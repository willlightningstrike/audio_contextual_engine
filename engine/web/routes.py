"""Local HTTP API for staged uploads, jobs, batches, and exports."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from engine.domain import ContentContext, JobRequest
from engine.ingest.probe import (
    MediaProbe,
    ProbeError,
    RequestedStreamInvalid,
    choose_audio_stream,
    probe_source,
)
from engine.lifecycle import JobStage
from engine.store import JobNotFound, JobStore
from engine.web.sse import job_events
from engine.web.snapshots import job_snapshot
from engine.web.uploads import UploadError, UploadManager
from engine.standards import load_taxonomy


def probe_media(path: Path) -> MediaProbe:
    return probe_source(path)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathInput(_Input):
    path: str = Field(min_length=1)
    audio_stream_index: int | None = Field(default=None, ge=0)
    requested_language: str | None = None
    requested_context: ContentContext | None = None

    @field_validator("audio_stream_index", mode="before")
    @classmethod
    def _valid_stream_index(cls, value: object) -> object:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("audio_stream_index must be a non-negative integer")
        return value

    @field_validator("requested_language", mode="before")
    @classmethod
    def _english_only(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str) or value.strip().lower() not in {"en", "english"}:
            raise ValueError("only English language requests are supported")
        return "en"


class UploadJobInput(_Input):
    upload_id: str = Field(min_length=1)
    audio_stream_index: int | None = Field(default=None, ge=0)
    requested_language: str | None = None
    requested_context: ContentContext | None = None

    @field_validator("audio_stream_index", mode="before")
    @classmethod
    def _valid_stream_index(cls, value: object) -> object:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("audio_stream_index must be a non-negative integer")
        return value

    @field_validator("requested_language", mode="before")
    @classmethod
    def _english_only(cls, value: object) -> object:
        return PathInput._english_only(value)


class BatchInput(_Input):
    path: str = Field(min_length=1)


def build_router(*, store: JobStore, runner: object, settings: object) -> APIRouter:
    router = APIRouter(prefix="/api")
    uploads_root = Path(getattr(settings, "uploads_root", Path(settings.data_dir) / "uploads"))
    upload_manager = UploadManager(
        uploads_root, ttl_seconds=int(getattr(settings, "upload_ttl_seconds", 86_400))
    )
    max_upload_bytes = int(getattr(settings, "max_upload_bytes", 1_073_741_824))
    allowed_suffixes = frozenset(
        suffix.lower() if str(suffix).startswith(".") else f".{str(suffix).lower()}"
        for suffix in getattr(settings, "allowed_suffixes", {".wav", ".mp4", ".mov", ".m4a", ".mp3"})
    )
    allowed_roots = tuple(Path(root).expanduser().resolve(strict=True) for root in settings.allowed_roots)

    def safe_path(raw_path: str, *, directory: bool = False) -> Path:
        candidate = Path(raw_path).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
            info = resolved.stat()
            if directory and not stat.S_ISDIR(info.st_mode):
                raise HTTPException(status_code=422, detail="path must be a directory")
            if not directory and not stat.S_ISREG(info.st_mode):
                raise HTTPException(status_code=422, detail="path must be a regular file")
            if not any(resolved.is_relative_to(root) for root in allowed_roots):
                raise HTTPException(status_code=403, detail="path is outside configured media roots")
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail="path is unavailable") from exc
        return resolved

    async def probe_payload(path: Path, requested: int | None = None) -> dict[str, Any]:
        try:
            probe = await asyncio.to_thread(probe_media, path)
            selected = choose_audio_stream(probe, requested)
        except RequestedStreamInvalid as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ProbeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "path": str(path),
            "container_kind": probe.container_kind.value,
            "duration_seconds": probe.duration_seconds,
            "title": probe.source_metadata.title,
            "selected_audio_stream": selected.index,
            "audio_streams": [stream.model_dump(mode="json") for stream in probe.audio_streams],
        }

    async def create_record(path: Path, *, language: str | None, context: ContentContext | None, stream: int | None):
        try:
            probe = await asyncio.to_thread(probe_media, path)
            selected = choose_audio_stream(probe, stream)
            request = JobRequest(
                source=path,
                requested_language=language,
                requested_context=context,
                audio_stream_index=selected.index,
            )
        except (ProbeError, ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = store.create(request)
        return record

    @router.post("/uploads", status_code=201)
    async def stage_upload(request: Request, filename: str) -> dict[str, Any]:
        if Path(filename).suffix.lower() not in allowed_suffixes:
            raise HTTPException(status_code=415, detail="unsupported media file suffix")
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_upload_bytes:
                    raise HTTPException(status_code=413, detail="upload exceeds configured byte limit")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid Content-Length") from exc
        try:
            upload_manager.prepare()
            upload_id, target = upload_manager.create_staging_file(filename)
        except UploadError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        temporary = target.with_name(f".{target.name}.partial")
        total = 0
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="upload exceeds configured byte limit")
                    stream.write(chunk)
            if total == 0:
                raise HTTPException(status_code=400, detail="empty upload")
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            try:
                target.parent.rmdir()
            except OSError:
                pass
            raise
        payload = await probe_payload(target)
        return {"upload_id": upload_id, "filename": filename, "size_bytes": total, "probe": payload}

    @router.post("/media/probe-path")
    async def probe_path(body: PathInput) -> dict[str, Any]:
        path = safe_path(body.path)
        return await probe_payload(path, body.audio_stream_index)

    @router.post("/jobs/path", status_code=202)
    async def create_path_job(body: PathInput) -> dict[str, Any]:
        path = safe_path(body.path)
        record = await create_record(path, language=body.requested_language, context=body.requested_context, stream=body.audio_stream_index)
        await runner.enqueue(record.id)  # type: ignore[attr-defined]
        return record.model_dump(mode="json")

    @router.post("/jobs/upload", status_code=202)
    async def create_upload_job(body: UploadJobInput) -> dict[str, Any]:
        try:
            path = upload_manager.get_unclaimed(body.upload_id)
            await probe_payload(path, body.audio_stream_index)
            path = upload_manager.claim(body.upload_id)
        except UploadError as exc:
            status = 409 if "claimed" in str(exc) else 404
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except HTTPException:
            raise
        record = await create_record(path, language=body.requested_language, context=body.requested_context, stream=body.audio_stream_index)
        await runner.enqueue(record.id)  # type: ignore[attr-defined]
        return record.model_dump(mode="json")

    @router.post("/batches/path", status_code=202)
    async def create_path_batch(body: BatchInput) -> dict[str, Any]:
        root = safe_path(body.path, directory=True)
        from engine.batch import discover_batch

        media = discover_batch(root, allowed_suffixes=allowed_suffixes)
        if not media:
            raise HTTPException(status_code=422, detail="directory contains no supported media files")
        children = []
        for child in media:
            child_path = safe_path(str(child))
            children.append(store.create(JobRequest(source=child_path)))
        try:
            batch = store.create_batch(root, tuple(child.id for child in children))
        except Exception as exc:
            raise HTTPException(status_code=500, detail="could not persist batch") from exc
        await runner.enqueue(batch.id)  # type: ignore[attr-defined]
        return batch.model_dump(mode="json")

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        try:
            return job_snapshot(store, store.load(job_id))
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @router.get("/jobs/{job_id}/events")
    async def events(job_id: str, request: Request):
        try:
            store.load(job_id)
        except (JobNotFound, ValueError) as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        heartbeat = float(getattr(settings, "sse_heartbeat_seconds", 15.0))
        return StreamingResponse(
            job_events(store, job_id, heartbeat_seconds=heartbeat),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @router.post("/jobs/{job_id}/cancel", status_code=202)
    async def cancel_job(job_id: str) -> dict[str, str]:
        try:
            store.load(job_id)
        except (JobNotFound, ValueError) as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        await runner.cancel(job_id)  # type: ignore[attr-defined]
        return {"id": job_id, "status": "cancellation_requested"}

    @router.get("/jobs/{job_id}/export")
    async def export_job(job_id: str) -> Response:
        try:
            contents = store.read_export(job_id)
        except (JobNotFound, ValueError) as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if contents is None:
            raise HTTPException(status_code=409, detail="export is available only after completion")
        return Response(contents, media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{job_id}.json"'})

    @router.get("/batches/{batch_id}")
    async def get_batch(batch_id: str) -> dict[str, Any]:
        try:
            return store.load_batch(batch_id).model_dump(mode="json")
        except (JobNotFound, ValueError) as exc:
            raise HTTPException(status_code=404, detail="batch not found") from exc

    @router.get("/taxonomy")
    async def taxonomy() -> list[dict[str, Any]]:
        return [
            {"id": row.id, "name": row.name, "path": list(row.path)}
            for row in load_taxonomy()
        ]

    @router.post("/batches/{batch_id}/cancel", status_code=202)
    async def cancel_batch(batch_id: str) -> dict[str, str]:
        try:
            store.load_batch(batch_id)
        except (JobNotFound, ValueError) as exc:
            raise HTTPException(status_code=404, detail="batch not found") from exc
        await runner.cancel(batch_id)  # type: ignore[attr-defined]
        return {"id": batch_id, "status": "cancellation_requested"}

    return router

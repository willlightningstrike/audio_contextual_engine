"""Server-sent progress snapshots read from durable storage."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from engine.lifecycle import JobStage
from engine.web.snapshots import job_snapshot


_TERMINAL = {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}


async def job_events(store: object, job_id: str, *, heartbeat_seconds: float = 15.0) -> AsyncIterator[str]:
    """Emit the current durable revision immediately, then only newer revisions."""
    last_revision = -1
    last_sent = asyncio.get_running_loop().time()
    while True:
        record = await asyncio.to_thread(store.load, job_id)  # type: ignore[attr-defined]
        revision = record.progress.revision
        if revision > last_revision:
            snapshot = await asyncio.to_thread(job_snapshot, store, record)
            payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
            yield f"id: {revision}\ndata: {payload}\n\n"
            last_revision = revision
            last_sent = asyncio.get_running_loop().time()
        if record.status in _TERMINAL:
            yield "event: terminal\ndata: {}\n\n"
            return
        now = asyncio.get_running_loop().time()
        if now - last_sent >= heartbeat_seconds:
            yield ": heartbeat\n\n"
            last_sent = now
        await asyncio.sleep(min(0.1, heartbeat_seconds, 0.1))

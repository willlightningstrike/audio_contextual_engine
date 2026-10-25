"""Private, bounded upload staging with single-use claims."""

from __future__ import annotations

from datetime import datetime, UTC
import os
from pathlib import Path
import stat
import threading
from uuid import UUID, uuid4


class UploadError(ValueError):
    """An upload is missing, expired, unsafe, or already claimed."""


class UploadManager:
    def __init__(self, root: Path, *, ttl_seconds: int = 86_400) -> None:
        self.root = Path(root)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

    def stage_path(self, upload_id: str, filename: str) -> Path:
        upload_id = self._id(upload_id)
        if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise UploadError("filename must be a plain file name")
        return self._safe_root() / upload_id / filename

    def prepare(self) -> None:
        with self._lock:
            root = self._safe_root(create=True)
            self._cleanup_locked(root)

    def create_staging_file(self, filename: str) -> tuple[str, Path]:
        if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise UploadError("filename must be a plain file name")
        with self._lock:
            root = self._safe_root(create=True)
            self._cleanup_locked(root)
            upload_id = str(uuid4())
            directory = root / upload_id
            directory.mkdir(mode=0o700)
            return upload_id, self.stage_path(upload_id, filename)

    def claim(self, upload_id: str) -> Path:
        with self._lock:
            self._cleanup_locked(self._safe_root(create=True))
            path = self._unclaimed_path_locked(upload_id)
            try:
                marker = path / ".claimed"
                fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(fd)
                return self._media_file(path)
            except FileExistsError as exc:
                raise UploadError("upload has already been claimed") from exc

    def get_unclaimed(self, upload_id: str) -> Path:
        """Inspect staged media without consuming its single-use claim."""
        with self._lock:
            self._cleanup_locked(self._safe_root(create=True))
            directory = self._unclaimed_path_locked(upload_id)
            return self._media_file(directory)

    def _unclaimed_path_locked(self, upload_id: str) -> Path:
        directory = self.stage_path(upload_id, "placeholder").parent
        try:
            info = directory.lstat()
        except FileNotFoundError as exc:
            raise UploadError("upload is missing or expired") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise UploadError("upload is unavailable")
        if (directory / ".claimed").exists():
            raise UploadError("upload has already been claimed")
        return directory

    @staticmethod
    def _media_file(directory: Path) -> Path:
        # Skip only staging's own files, not user names that start with a dot.
        names = [
            name for name in os.listdir(directory)
            if name != ".claimed" and not (name.startswith(".") and name.endswith(".partial"))
        ]
        if len(names) != 1:
            raise UploadError("upload is unavailable")
        source = directory / names[0]
        try:
            info = source.lstat()
        except FileNotFoundError as exc:
            raise UploadError("upload is unavailable") from exc
        if not stat.S_ISREG(info.st_mode):
            raise UploadError("upload is unavailable")
        return source

    def release_failed_claim(self, path: Path) -> None:
        """Keep failed creations non-reusable; marker is intentionally durable."""
        del path

    def _cleanup_locked(self, root: Path) -> None:
        now = datetime.now(UTC).timestamp()
        for entry in root.iterdir():
            try:
                info = entry.lstat()
                if not stat.S_ISDIR(info.st_mode):
                    continue
                if os.path.lexists(entry / ".claimed"):
                    continue
                if now - info.st_mtime <= self.ttl_seconds:
                    continue
                for child in entry.iterdir():
                    child_info = child.lstat()
                    if stat.S_ISREG(child_info.st_mode) or stat.S_ISLNK(child_info.st_mode):
                        child.unlink()
                entry.rmdir()
            except (FileNotFoundError, NotADirectoryError, OSError):
                continue

    def _safe_root(self, *, create: bool = False) -> Path:
        root = self.root.expanduser().absolute()
        if create:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            info = root.lstat()
        except FileNotFoundError:
            raise UploadError("upload storage is unavailable")
        if not stat.S_ISDIR(info.st_mode):
            raise UploadError("upload storage must be a real directory")
        return root

    @staticmethod
    def _id(upload_id: str) -> str:
        try:
            canonical = str(UUID(upload_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise UploadError("invalid upload ID") from exc
        if canonical != upload_id:
            raise UploadError("invalid upload ID")
        return canonical

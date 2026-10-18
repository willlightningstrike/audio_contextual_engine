"""Local application settings for the single-user macOS service."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from collections.abc import Mapping


DEFAULT_UPLOAD_BYTES = 1_073_741_824
DEFAULT_MEDIA_SUFFIXES = frozenset(
    {".wav", ".mp4", ".mov", ".m4a", ".mp3", ".mkv", ".webm"}
)
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def default_data_directory() -> Path:
    return Path.home() / "Library" / "Application Support" / "Contextual Engine"


@dataclass(frozen=True)
class Settings:
    """Filesystem and loopback-server configuration; no global HOME mutation."""

    data_dir: Path = field(default_factory=default_data_directory)
    allowed_roots: tuple[Path, ...] = field(default_factory=lambda: (Path.home(),))
    max_upload_bytes: int = DEFAULT_UPLOAD_BYTES
    allowed_suffixes: frozenset[str] = DEFAULT_MEDIA_SUFFIXES
    upload_ttl_seconds: int = 86_400
    sse_heartbeat_seconds: float = 15.0
    bind_host: str = DEFAULT_BIND_HOST
    port: int = DEFAULT_PORT

    @classmethod
    def from_environment(
        cls, *, environ: Mapping[str, str] | None = None
    ) -> "Settings":
        values = os.environ if environ is None else environ
        raw_data_dir = values.get("CONTEXTUAL_ENGINE_DATA_DIR")
        data_dir = (
            Path(raw_data_dir).expanduser()
            if raw_data_dir
            else default_data_directory()
        )
        return cls(data_dir=data_dir)

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def uploads_root(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def taxonomy_index_dir(self) -> Path:
        return self.models_dir / "taxonomy-index"

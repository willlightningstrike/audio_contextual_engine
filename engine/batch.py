"""Safe, deterministic discovery for non-recursive media batches."""

from __future__ import annotations

from pathlib import Path
import stat


def discover_batch(
    directory: Path, *, allowed_suffixes: frozenset[str]
) -> tuple[Path, ...]:
    """Return sorted immediate regular files whose resolved target stays in the root.

    Symbolic links are rejected rather than followed, avoiding escaped paths,
    aliases, and retargeting between discovery and later use.
    """
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    suffixes = frozenset(
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in allowed_suffixes
        if suffix and suffix != "."
    )
    accepted: list[Path] = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        if candidate.suffix.lower() not in suffixes:
            continue
        try:
            entry_info = candidate.lstat()
            if not stat.S_ISREG(entry_info.st_mode):
                continue
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            if stat.S_ISREG(resolved.stat().st_mode):
                accepted.append(candidate)
        except (OSError, ValueError):
            continue
    return tuple(accepted)

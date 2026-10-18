"""Offline, pinned IAB and OpenRTB standards used by the engine."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


CATTAX_CONTENT_3_0 = 7
CONTEXT_VIDEO = 1
CONTEXT_MUSIC = 3
CONTEXT_OTHER = 6
CONTEXT_UNKNOWN = 7

_TAXONOMY_RESOURCE = files("engine").joinpath(
    "data", "standards", "content-taxonomy-3.0.tsv"
)
_PATH_FIELDS = ("Tier 1", "Tier 2", "Tier 3", "Tier 4")


@dataclass(frozen=True)
class TaxonomyRow:
    id: str
    name: str
    tier: int
    parent_id: str | None
    path: tuple[str, ...]


@lru_cache(maxsize=1)
def load_taxonomy() -> tuple[TaxonomyRow, ...]:
    """Load the vendored IAB Content Taxonomy 3.0 without network access."""
    taxonomy_resource = _TAXONOMY_RESOURCE
    if not taxonomy_resource.is_file():
        taxonomy_resource = files("data").joinpath(
            "standards", "content-taxonomy-3.0.tsv"
        )
    with taxonomy_resource.open("r", newline="", encoding="utf-8-sig") as taxonomy_file:
        next(taxonomy_file)
        reader = csv.DictReader(taxonomy_file, delimiter="\t")
        rows = []
        for source_row in reader:
            taxonomy_id = source_row["Unique ID"]
            parent_id = source_row["Parent"] or None
            if parent_id == taxonomy_id:
                parent_id = None
            path = tuple(source_row[field] for field in _PATH_FIELDS if source_row[field])
            rows.append(
                TaxonomyRow(
                    id=taxonomy_id,
                    name=source_row["Name"],
                    tier=len(path),
                    parent_id=parent_id,
                    path=path,
                )
            )
    return tuple(rows)

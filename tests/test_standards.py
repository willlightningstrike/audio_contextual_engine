import hashlib
import json
from pathlib import Path

from engine.standards import (
    CATTAX_CONTENT_3_0,
    CONTEXT_MUSIC,
    CONTEXT_OTHER,
    CONTEXT_UNKNOWN,
    CONTEXT_VIDEO,
    load_taxonomy,
)


ROOT = Path(__file__).resolve().parents[1]
TSV_PATH = ROOT / "data" / "standards" / "content-taxonomy-3.0.tsv"
MANIFEST_PATH = ROOT / "data" / "standards" / "manifest.json"
EXPECTED_SHA256 = "88a70b83feca05b7c9f7ed6f6a70ac501b863eff7c1aca83fb90356e86e9492e"
EXPECTED_COMMIT = "0c236b7d73811983bc0e7a314ebb2769acde2291"


def test_adcom_constants_are_pinned():
    assert CATTAX_CONTENT_3_0 == 7
    assert (CONTEXT_VIDEO, CONTEXT_MUSIC, CONTEXT_OTHER, CONTEXT_UNKNOWN) == (1, 3, 6, 7)
    assert CATTAX_CONTENT_3_0 != 8


def test_taxonomy_has_unique_ids_and_preserves_numeric_ids_as_strings():
    taxonomy = load_taxonomy()
    ids = [row.id for row in taxonomy]
    assert len(ids) >= 700
    assert len(ids) == len(set(ids))
    assert all(isinstance(item, str) for item in ids)
    assert next(item for item in ids if item == "150").isdigit()


def test_taxonomy_normalizes_self_parented_top_level_rows_and_freezes_rows():
    taxonomy = load_taxonomy()
    top_level = next(row for row in taxonomy if row.id == "150")
    assert top_level.parent_id is None
    assert top_level.path == ("Attractions",)
    assert taxonomy[0].__dataclass_params__.frozen is True


def test_vendored_bytes_and_provenance_are_pinned():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert hashlib.sha256(TSV_PATH.read_bytes()).hexdigest() == EXPECTED_SHA256
    assert manifest["iab_content_taxonomy_version"] == "3.0"
    assert manifest["iab_taxonomy_tag"] == "Taxos-202308"
    assert manifest["iab_taxonomy_commit"] == EXPECTED_COMMIT
    assert manifest["sha256"] == EXPECTED_SHA256

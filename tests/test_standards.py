import hashlib
import json
from pathlib import Path

from engine.standards import (
    CATTAX_CONTENT_3_1,
    CONTEXT_MUSIC,
    CONTEXT_OTHER,
    CONTEXT_UNKNOWN,
    CONTEXT_VIDEO,
    load_taxonomy,
)


ROOT = Path(__file__).resolve().parents[1]
TSV_PATH = ROOT / "data" / "standards" / "content-taxonomy-3.1.tsv"
MANIFEST_PATH = ROOT / "data" / "standards" / "manifest.json"
EXPECTED_SHA256 = "7212cdc496ba347a03e703b1932bdcdd4fd29089b058f4edeb4d3da1f1222ea7"
EXPECTED_COMMIT = "6dc67c743572e856c7fa86d310cfa844c9fcda85"


def test_adcom_constants_are_pinned():
    assert CATTAX_CONTENT_3_1 == 9
    assert (CONTEXT_VIDEO, CONTEXT_MUSIC, CONTEXT_OTHER, CONTEXT_UNKNOWN) == (1, 3, 6, 7)


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
    assert manifest["iab_content_taxonomy_version"] == "3.1"
    assert manifest["iab_taxonomy_commit"] == EXPECTED_COMMIT
    assert manifest["openrtb_cattax_content_3_1"] == 9
    assert manifest["sha256"] == EXPECTED_SHA256
    assert "iab_taxonomy_tag" not in manifest and "openrtb_cattax_content_3_0" not in manifest


def test_taxonomy_3_1_rows():
    rows = {row.id: row for row in load_taxonomy()}
    assert len(rows) == 704
    assert rows["700"].path == ("Genres", "Thriller")
    assert rows["338"].path == ("Entertainment", "Music")
    assert rows["342"].path == ("Entertainment", "Music", "Adult Album Alternative")
    assert rows["160"].name == "Parks & Nature"
    assert not any(row.path[:1] == ("Music",) for row in rows.values())

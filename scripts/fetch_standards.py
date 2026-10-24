"""Fetch and pin the IAB Content Taxonomy 3.1 for offline production use."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


# Last commit touching the 3.1 file; newer than the Taxos-202412 tag, which was later corrected.
COMMIT = "6dc67c743572e856c7fa86d310cfa844c9fcda85"
REPOSITORY = "https://github.com/InteractiveAdvertisingBureau/Taxonomies"
UPSTREAM_PATH = "Content Taxonomies/Content Taxonomy 3.1.tsv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/InteractiveAdvertisingBureau/Taxonomies/"
    f"{COMMIT}/Content%20Taxonomies/Content%20Taxonomy%203.1.tsv"
)
EXPECTED_SHA256 = "7212cdc496ba347a03e703b1932bdcdd4fd29089b058f4edeb4d3da1f1222ea7"

ROOT = Path(__file__).resolve().parents[1]
STANDARDS_DIR = ROOT / "data" / "standards"
TSV_PATH = STANDARDS_DIR / "content-taxonomy-3.1.tsv"
MANIFEST_PATH = STANDARDS_DIR / "manifest.json"


def main() -> None:
    with urlopen(SOURCE_URL) as response:
        taxonomy_bytes = response.read()
    actual_sha256 = hashlib.sha256(taxonomy_bytes).hexdigest()
    if actual_sha256 != EXPECTED_SHA256:
        raise SystemExit(
            f"taxonomy SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
        )

    STANDARDS_DIR.mkdir(parents=True, exist_ok=True)
    TSV_PATH.write_bytes(taxonomy_bytes)
    manifest = {
        "iab_content_taxonomy_version": "3.1",
        "iab_taxonomy_commit": COMMIT,
        "repository": REPOSITORY,
        "upstream_path": UPSTREAM_PATH,
        "source_url": SOURCE_URL,
        "sha256": EXPECTED_SHA256,
        "openrtb_cattax_content_3_1": 9,
        "context_constants": {"video": 1, "music": 3, "other": 6, "unknown": 7},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Verified {TSV_PATH} SHA-256: {actual_sha256}")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

"""Fetch and pin the IAB Content Taxonomy 3.0 for offline production use."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


COMMIT = "0c236b7d73811983bc0e7a314ebb2769acde2291"
TAG = "Taxos-202308"
REPOSITORY = "https://github.com/InteractiveAdvertisingBureau/Taxonomies"
UPSTREAM_PATH = "Content Taxonomies/Content Taxonomy 3.0.tsv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/InteractiveAdvertisingBureau/Taxonomies/"
    f"{COMMIT}/Content%20Taxonomies/Content%20Taxonomy%203.0.tsv"
)
EXPECTED_SHA256 = "88a70b83feca05b7c9f7ed6f6a70ac501b863eff7c1aca83fb90356e86e9492e"

ROOT = Path(__file__).resolve().parents[1]
STANDARDS_DIR = ROOT / "data" / "standards"
TSV_PATH = STANDARDS_DIR / "content-taxonomy-3.0.tsv"
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
        "iab_content_taxonomy_version": "3.0",
        "iab_taxonomy_tag": TAG,
        "iab_taxonomy_commit": COMMIT,
        "repository": REPOSITORY,
        "upstream_path": UPSTREAM_PATH,
        "source_url": SOURCE_URL,
        "sha256": EXPECTED_SHA256,
        "openrtb_cattax_content_3_0": 7,
        "context_constants": {"video": 1, "music": 3, "other": 6, "unknown": 7},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Verified {TSV_PATH} SHA-256: {actual_sha256}")
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

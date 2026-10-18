from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.standards import TaxonomyRow


class FixedEmbedder:
    def __init__(self, responses: list[np.ndarray]) -> None:
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return self.responses.pop(0)


def _rows(*ids: str) -> tuple[TaxonomyRow, ...]:
    return tuple(
        TaxonomyRow(
            id=taxonomy_id,
            name=f"Name {taxonomy_id}",
            tier=2,
            parent_id="root",
            path=("Root", f"Name {taxonomy_id}"),
        )
        for taxonomy_id in ids
    )


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shortlist_breaks_actual_similarity_ties_by_ascending_string_id():
    from engine.signals.taxonomy import TaxonomyIndex

    rows = _rows("80DV8O", "633", "52")
    embedder = FixedEmbedder([np.array([[7.0, 0.0]], dtype=np.float32)])
    index = TaxonomyIndex(
        rows,
        np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        embedder,
    )

    candidates = index.shortlist("equal relevance", top_k=3, token=CancellationToken())

    assert [candidate.taxonomy_id for candidate in candidates] == ["52", "633", "80DV8O"]
    assert [candidate.similarity for candidate in candidates] == [1.0, 1.0, 1.0]


def test_shortlist_uses_frozen_query_prefix_and_preserves_original_cosines():
    from engine.signals.taxonomy import QUERY_PREFIX, TaxonomyIndex

    rows = _rows("a", "b")
    embedder = FixedEmbedder([np.array([[4.0, 0.0]], dtype=np.float32)])
    index = TaxonomyIndex(
        rows,
        np.array([[3.0, 4.0], [2.0, 0.0]], dtype=np.float32),
        embedder,
    )

    candidates = index.shortlist("camera transcript", top_k=2, token=CancellationToken())

    assert embedder.calls == [[QUERY_PREFIX + "camera transcript"]]
    assert [candidate.taxonomy_id for candidate in candidates] == ["b", "a"]
    assert candidates[0].similarity == pytest.approx(1.0)
    assert candidates[1].similarity == pytest.approx(0.6)
    assert candidates[1].name == "Name a"
    assert candidates[1].path == ("Root", "Name a")


@pytest.mark.parametrize("query", ["", "  \n"])
def test_shortlist_rejects_empty_query_without_encoding(query: str):
    from engine.signals.taxonomy import TaxonomyIndex

    embedder = FixedEmbedder([])
    index = TaxonomyIndex(_rows("1"), np.array([[1.0]], dtype=np.float32), embedder)

    with pytest.raises(ValueError, match="non-empty"):
        index.shortlist(query, token=CancellationToken())
    assert embedder.calls == []


@pytest.mark.parametrize("top_k", [0, -1, 3, 1.5, True])
def test_shortlist_rejects_top_k_outside_index_bounds(top_k):
    from engine.signals.taxonomy import TaxonomyIndex

    index = TaxonomyIndex(_rows("1", "2"), np.eye(2, dtype=np.float32), FixedEmbedder([]))

    with pytest.raises(ValueError, match="top_k"):
        index.shortlist("query", top_k=top_k, token=CancellationToken())


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[np.nan, 0.0]], dtype=np.float32),
        np.array([[np.inf, 0.0]], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    ],
)
def test_index_rejects_zero_nonfinite_or_mismatched_vectors(vectors: np.ndarray):
    from engine.signals.taxonomy import TaxonomyIndex

    with pytest.raises(ValueError, match="vector|matrix"):
        TaxonomyIndex(_rows("1"), vectors, FixedEmbedder([]))


def test_index_rejects_complex_stored_vectors() -> None:
    from engine.signals.taxonomy import TaxonomyIndex

    with pytest.raises(ValueError, match="taxonomy vector matrix"):
        TaxonomyIndex(
            _rows("1"),
            np.array([[1.0 + 2.0j]], dtype=np.complex64),
            FixedEmbedder([]),
        )


@pytest.mark.parametrize(
    "query_vector",
    [
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[np.nan, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    ],
)
def test_shortlist_rejects_invalid_query_vectors(query_vector: np.ndarray):
    from engine.signals.taxonomy import TaxonomyIndex

    index = TaxonomyIndex(
        _rows("1"),
        np.array([[1.0, 0.0]], dtype=np.float32),
        FixedEmbedder([query_vector]),
    )

    with pytest.raises(ValueError, match="query embedding"):
        index.shortlist("query", top_k=1, token=CancellationToken())


def test_shortlist_rejects_complex_query_vectors() -> None:
    from engine.signals.taxonomy import TaxonomyIndex

    index = TaxonomyIndex(
        _rows("1"),
        np.array([[1.0]], dtype=np.float32),
        FixedEmbedder([np.array([[1.0 + 2.0j]], dtype=np.complex64)]),
    )

    with pytest.raises(ValueError, match="query embedding matrix"):
        index.shortlist("query", top_k=1, token=CancellationToken())


def test_shortlist_checks_cancellation_before_and_after_embedding():
    from engine.signals.taxonomy import TaxonomyIndex

    rows = _rows("1")
    initial = CancellationToken()
    initial.cancel()
    never_called = FixedEmbedder([])
    index = TaxonomyIndex(rows, np.array([[1.0]], dtype=np.float32), never_called)
    with pytest.raises(JobCancelled):
        index.shortlist("query", top_k=1, token=initial)
    assert never_called.calls == []

    cancelled_during_encode = CancellationToken()

    class CancellingEmbedder:
        def encode(self, texts):
            cancelled_during_encode.cancel()
            return np.array([[1.0]], dtype=np.float32)

    index = TaxonomyIndex(rows, np.array([[1.0]], dtype=np.float32), CancellingEmbedder())
    with pytest.raises(JobCancelled):
        index.shortlist("query", top_k=1, token=cancelled_during_encode)


def test_shortlist_checks_cancellation_after_matrix_search():
    from engine.signals.taxonomy import TaxonomyIndex

    class FourthBoundaryCancellation:
        def __init__(self) -> None:
            self.calls = 0

        def raise_if_cancelled(self) -> None:
            self.calls += 1
            if self.calls == 4:
                raise JobCancelled

    token = FourthBoundaryCancellation()
    index = TaxonomyIndex(
        _rows("1"),
        np.array([[1.0]], dtype=np.float32),
        FixedEmbedder([np.array([[1.0]], dtype=np.float32)]),
    )

    with pytest.raises(JobCancelled):
        index.shortlist("query", top_k=1, token=token)
    assert token.calls == 4


def test_canonical_taxonomy_text_matches_the_reviewed_trial_exactly():
    from engine.signals.taxonomy import CANONICALIZATION_VERSION, canonical_taxonomy_text

    row = TaxonomyRow("601", "Computer Peripherals", 4, "600", ("Technology", "Computing", "Hardware", "Computer Peripherals"))

    assert canonical_taxonomy_text(row) == (
        "Category: Computer Peripherals\n"
        "Hierarchy: Technology > Computing > Hardware > Computer Peripherals"
    )
    assert CANONICALIZATION_VERSION == "iab-name-hierarchy-v1"


def test_build_and_load_round_trip_validates_identity_and_artifacts(tmp_path: Path):
    from engine.signals.taxonomy import (
        build_taxonomy_index,
        load_taxonomy_index,
    )

    taxonomy_path = tmp_path / "taxonomy.tsv"
    taxonomy_path.write_bytes(b"reviewed taxonomy bytes\n")
    rows = _rows("80DV8O", "52")
    build_embedder = FixedEmbedder(
        [np.array([[3.0, 4.0], [0.0, 8.0]], dtype=np.float64)]
    )
    output = tmp_path / "index"

    manifest = build_taxonomy_index(
        output,
        taxonomy_path=taxonomy_path,
        taxonomy=rows,
        embedder=build_embedder,
        embedding_revision="revision-1",
    )

    assert build_embedder.calls == [[
        "Category: Name 80DV8O\nHierarchy: Root > Name 80DV8O",
        "Category: Name 52\nHierarchy: Root > Name 52",
    ]]
    assert manifest["ordered_ids"] == ["80DV8O", "52"]
    assert manifest["taxonomy_sha256"] == hashlib.sha256(taxonomy_path.read_bytes()).hexdigest()
    assert manifest["embedding_revision"] == "revision-1"
    assert manifest["canonicalization_version"] == "iab-name-hierarchy-v1"
    assert manifest["vectors"]["dtype"] == "float32"
    assert manifest["vectors"]["shape"] == [2, 2]
    assert not list(output.glob("*.tmp"))

    loaded = load_taxonomy_index(
        output,
        taxonomy_path=taxonomy_path,
        taxonomy=rows,
        embedder=FixedEmbedder([np.array([[1.0, 0.0]], dtype=np.float32)]),
        expected_embedding_revision="revision-1",
    )
    result = loaded.shortlist("query", top_k=2, token=CancellationToken())
    assert [(item.taxonomy_id, item.similarity) for item in result] == [
        ("80DV8O", pytest.approx(0.6)),
        ("52", pytest.approx(0.0)),
    ]


def test_build_rejects_symlink_output_without_touching_existing_artifacts(tmp_path: Path):
    from engine.signals.taxonomy import build_taxonomy_index

    taxonomy_path = tmp_path / "taxonomy.tsv"
    taxonomy_path.write_bytes(b"taxonomy")
    existing = tmp_path / "existing-index"
    existing.mkdir()
    (existing / "manifest.json").write_bytes(b"existing manifest")
    (existing / "taxonomy-vectors-existing.npy").write_bytes(b"existing vectors")
    before = {path.name: path.read_bytes() for path in existing.iterdir()}
    linked_output = tmp_path / "linked-index"
    linked_output.symlink_to(existing, target_is_directory=True)
    embedder = FixedEmbedder([np.array([[1.0]], dtype=np.float32)])

    with pytest.raises(ValueError, match="symlink"):
        build_taxonomy_index(
            linked_output,
            taxonomy_path=taxonomy_path,
            taxonomy=_rows("52"),
            embedder=embedder,
            embedding_revision="revision-1",
        )

    assert {path.name: path.read_bytes() for path in existing.iterdir()} == before
    assert embedder.calls == []


def test_load_rejects_symlink_index_without_touching_existing_artifacts(tmp_path: Path):
    from engine.signals.taxonomy import build_taxonomy_index, load_taxonomy_index

    taxonomy_path = tmp_path / "taxonomy.tsv"
    taxonomy_path.write_bytes(b"taxonomy")
    existing = tmp_path / "existing-index"
    rows = _rows("52")
    build_taxonomy_index(
        existing,
        taxonomy_path=taxonomy_path,
        taxonomy=rows,
        embedder=FixedEmbedder([np.array([[1.0]], dtype=np.float32)]),
        embedding_revision="revision-1",
    )
    before = {path.name: path.read_bytes() for path in existing.iterdir()}
    linked_index = tmp_path / "linked-index"
    linked_index.symlink_to(existing, target_is_directory=True)
    query_embedder = FixedEmbedder([])

    with pytest.raises(ValueError, match="symlink"):
        load_taxonomy_index(
            linked_index,
            taxonomy_path=taxonomy_path,
            taxonomy=rows,
            embedder=query_embedder,
            expected_embedding_revision="revision-1",
        )

    assert {path.name: path.read_bytes() for path in existing.iterdir()} == before
    assert query_embedder.calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest, taxonomy: manifest.update(embedding_revision="stale"), "embedding revision"),
        (lambda manifest, taxonomy: taxonomy.write_bytes(b"changed"), "taxonomy SHA"),
        (lambda manifest, taxonomy: manifest.update(ordered_ids=["wrong"]), "ordered IDs"),
    ],
)
def test_load_rejects_stale_manifest_before_search(tmp_path: Path, mutation, message: str):
    from engine.signals.taxonomy import build_taxonomy_index, load_taxonomy_index

    taxonomy_path = tmp_path / "taxonomy.tsv"
    taxonomy_path.write_bytes(b"taxonomy")
    output = tmp_path / "index"
    rows = _rows("52")
    build_taxonomy_index(
        output,
        taxonomy_path=taxonomy_path,
        taxonomy=rows,
        embedder=FixedEmbedder([np.array([[1.0]], dtype=np.float32)]),
        embedding_revision="revision-1",
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutation(manifest, taxonomy_path)
    manifest_path.write_text(json.dumps(manifest))
    query_embedder = FixedEmbedder([])

    with pytest.raises(ValueError, match=message):
        load_taxonomy_index(
            output,
            taxonomy_path=taxonomy_path,
            taxonomy=rows,
            embedder=query_embedder,
            expected_embedding_revision="revision-1",
        )
    assert query_embedder.calls == []


def test_load_rejects_modified_vector_artifact_without_pickle(tmp_path: Path, monkeypatch):
    import engine.signals.taxonomy as taxonomy_module

    taxonomy_path = tmp_path / "taxonomy.tsv"
    taxonomy_path.write_bytes(b"taxonomy")
    output = tmp_path / "index"
    rows = _rows("52")
    manifest = taxonomy_module.build_taxonomy_index(
        output,
        taxonomy_path=taxonomy_path,
        taxonomy=rows,
        embedder=FixedEmbedder([np.array([[1.0]], dtype=np.float32)]),
        embedding_revision="revision-1",
    )
    vector_path = output / manifest["vectors"]["path"]
    vector_path.write_bytes(vector_path.read_bytes() + b"tampered")
    original_load = np.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(taxonomy_module.np, "load", recording_load)
    with pytest.raises(ValueError, match="vector artifact"):
        taxonomy_module.load_taxonomy_index(
            output,
            taxonomy_path=taxonomy_path,
            taxonomy=rows,
            embedder=FixedEmbedder([]),
            expected_embedding_revision="revision-1",
        )
    assert calls == []


def test_embedding_only_manifest_verification_ignores_obsolete_model_roles(tmp_path: Path):
    from engine.models.registry import MODEL_SPECS, load_verified_embedding_path

    snapshot = (tmp_path / "embeddings").resolve()
    snapshot.mkdir()
    weights = b"embedding weights"
    (snapshot / "model.safetensors").write_bytes(weights)
    manifest = {
        "schema_version": 1,
        "models": {
            "embeddings": {
                **MODEL_SPECS["embeddings"],
                "local_path": str(snapshot),
                "files": [{
                    "path": "model.safetensors",
                    "bytes": len(weights),
                    "sha256": hashlib.sha256(weights).hexdigest(),
                }],
            },
            "obsolete": {
                "local_path": str(tmp_path / "deleted"),
                "files": [],
            },
        },
    }
    manifest_path = tmp_path / "model-manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    assert load_verified_embedding_path(manifest_path) == snapshot


def test_embedding_only_manifest_rejects_non_object_payload(tmp_path: Path):
    from engine.models.registry import load_verified_embedding_path

    manifest_path = tmp_path / "model-manifest.json"
    manifest_path.write_text("[]")

    with pytest.raises(ValueError, match="invalid model manifest"):
        load_verified_embedding_path(manifest_path)


def test_diagnostic_replay_records_full_candidates_and_stable_repeat_ordering():
    from engine.signals.taxonomy import TaxonomyIndex

    build_script = _load_script("build_taxonomy_index")
    index = TaxonomyIndex(
        _rows("80DV8O", "52"),
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        FixedEmbedder(
            [
                np.array([[1.0, 0.0]], dtype=np.float32),
                np.array([[1.0, 0.0]], dtype=np.float32),
            ]
        ),
    )

    evidence = build_script.run_diagnostics(
        index,
        [{"clip_id": "N01", "transcript": "camera transcript"}],
        top_k=2,
    )

    assert evidence == [
        {
            "clip_id": "N01",
            "candidate_count": 2,
            "stable_repeat_ordering": True,
            "candidates": [
                {
                    "taxonomy_id": "52",
                    "name": "Name 52",
                    "path": ["Root", "Name 52"],
                    "similarity": 1.0,
                },
                {
                    "taxonomy_id": "80DV8O",
                    "name": "Name 80DV8O",
                    "path": ["Root", "Name 80DV8O"],
                    "similarity": 1.0,
                },
            ],
        }
    ]


def test_build_script_rejects_noncanonical_taxonomy_before_touching_index(
    tmp_path: Path, monkeypatch
):
    build_script = _load_script("build_taxonomy_index")
    custom_taxonomy = tmp_path / "custom-taxonomy.tsv"
    custom_taxonomy.write_bytes(b"different taxonomy bytes")
    index_dir = tmp_path / "existing-index"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_bytes(b"existing manifest")
    (index_dir / "taxonomy-vectors-existing.npy").write_bytes(b"existing vectors")
    evidence_path = index_dir / "build-evidence.json"
    evidence_path.write_bytes(b"existing evidence")
    before = {path.name: path.read_bytes() for path in index_dir.iterdir()}
    monkeypatch.setattr(build_script, "_swap_bytes", lambda: 0)

    with pytest.raises(ValueError, match="canonical vendored taxonomy"):
        build_script.build_and_diagnose(
            model_manifest=tmp_path / "missing-model-manifest.json",
            taxonomy_path=custom_taxonomy,
            index_dir=index_dir,
            evidence_path=evidence_path,
            diagnostic_sources=(),
        )

    assert {path.name: path.read_bytes() for path in index_dir.iterdir()} == before

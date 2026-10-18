from __future__ import annotations

from pathlib import Path

from engine.batch import discover_batch


def test_batch_discovery_is_sorted_non_recursive_and_suffix_filtered(tmp_path: Path):
    root = tmp_path / "media"
    root.mkdir()
    (root / "z.wav").write_bytes(b"z")
    (root / "a.MP4").write_bytes(b"a")
    (root / "ignored.txt").write_bytes(b"x")
    nested = root / "nested"
    nested.mkdir()
    (nested / "nested.wav").write_bytes(b"n")

    assert discover_batch(root, allowed_suffixes=frozenset({".wav", ".mp4"})) == (
        root / "a.MP4",
        root / "z.wav",
    )


def test_batch_discovery_rejects_symlink_that_resolves_outside_allowed_root(tmp_path: Path):
    root = tmp_path / "media"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"secret")
    (root / "linked.wav").symlink_to(outside)

    assert discover_batch(root, allowed_suffixes=frozenset({".wav"})) == ()


def test_batch_discovery_rejects_in_root_file_symlinks_to_avoid_aliasing(tmp_path: Path):
    root = tmp_path / "media"
    root.mkdir()
    target = root / "real.wav"
    target.write_bytes(b"audio")
    alias = root / "alias.wav"
    alias.symlink_to(target)

    assert discover_batch(root, allowed_suffixes=frozenset({".wav"})) == (target,)


def test_batch_snapshot_atomically_persists_child_outcomes_and_separate_counts(tmp_path: Path):
    from uuid import UUID

    from engine.domain import JobRequest
    from engine.lifecycle import JobStage
    from engine.store import JobStore

    ids = iter(
        UUID(f"00000000-0000-0000-0000-{number:012d}")
        for number in (1, 2, 3, 4)
    )
    store = JobStore(tmp_path / "jobs", uuid_factory=lambda: next(ids))
    children = tuple(store.create(JobRequest(source=f"{name}.wav")) for name in ("ok", "non-english", "instrumental"))
    batch = store.create_batch(tmp_path, tuple(child.id for child in children))
    assert batch.success_count == batch.failure_count == batch.cancelled_count == 0
    batch = batch.model_copy(
        update={
            "status": JobStage.BATCH_PROCESSING,
            "progress": batch.progress.advance(
                stage=JobStage.BATCH_PROCESSING, percent=0, message="processing"
            ),
        }
    )
    store.save_batch(batch)

    for child, outcome in zip(
        children,
        (JobStage.COMPLETE, JobStage.FAILED, JobStage.FAILED),
        strict=True,
    ):
        if outcome is JobStage.FAILED:
            progress = child.progress.advance(
                stage=JobStage.FAILED, percent=0, message="fixture failure"
            )
            store.save(
                child.model_copy(
                    update={"status": JobStage.FAILED, "progress": progress, "error": "fixture failure"}
                )
            )
            continue
        for stage in (
            JobStage.EXTRACTING_AUDIO,
            JobStage.DETECTING_LANGUAGE,
            JobStage.TRANSCRIBING,
            JobStage.ANALYSING,
            JobStage.EXPORTING,
        ):
            child = store.load(child.id)
            progress = child.progress.advance(stage=stage, percent=90, message=stage.value)
            child = child.model_copy(update={"status": stage, "progress": progress})
            store.save(child)
        store.promote_result(child.id, {"id": "fixture"})

    for child, outcome in zip(
        children,
        (JobStage.COMPLETE, JobStage.FAILED, JobStage.FAILED),
        strict=True,
    ):
        batch = batch.model_copy(
            update={
                "child_outcomes": {**batch.child_outcomes, child.id: outcome},
                "success_count": batch.success_count + (outcome is JobStage.COMPLETE),
                "failure_count": batch.failure_count + (outcome is JobStage.FAILED),
                "cancelled_count": batch.cancelled_count + (outcome is JobStage.CANCELLED),
                "active_child_job_id": None,
                "progress": batch.progress.advance(
                    stage=batch.status,
                    percent=10 * (len(batch.child_outcomes) + 1),
                    message="recorded child outcome",
                ),
            }
        )
        store.save_batch(batch)

    batch = batch.model_copy(
        update={
            "status": JobStage.COMPLETE,
            "progress": batch.progress.advance(
                stage=JobStage.COMPLETE, percent=100, message="batch processing complete"
            ),
        }
    )
    store.save_batch(batch)
    completed = store.load_batch(batch.id)
    assert completed.success_count == 1
    assert completed.failure_count == 2
    assert completed.child_outcomes[children[0].id] is JobStage.COMPLETE
    assert completed.all_children_succeeded is False


def test_batch_recovery_marks_processing_parent_failed_without_replaying_it(tmp_path: Path):
    from uuid import UUID

    from engine.domain import JobRequest
    from engine.lifecycle import JobStage
    from engine.store import JobStore

    ids = iter(UUID(f"00000000-0000-0000-0000-{number:012d}") for number in (11, 12))
    store = JobStore(tmp_path / "jobs", uuid_factory=lambda: next(ids))
    child = store.create(JobRequest(source="interrupted.wav"))
    batch = store.create_batch(tmp_path, (child.id,))
    processing = batch.progress.advance(
        stage=JobStage.BATCH_PROCESSING, percent=10, message="processing"
    )
    batch = batch.model_copy(update={"status": JobStage.BATCH_PROCESSING, "progress": processing})
    store.save_batch(batch)

    recovered = JobStore(store.jobs_root).recover_batches()

    assert len(recovered) == 1
    assert recovered[0].status is JobStage.FAILED
    assert recovered[0].error == "interrupted_by_restart"


def test_batch_cannot_persist_a_child_outcome_that_disagrees_with_its_job(tmp_path: Path):
    import pytest
    from uuid import UUID

    from engine.domain import JobRequest
    from engine.lifecycle import JobStage
    from engine.store import InvalidJobTransition, JobStore

    ids = iter(UUID(f"00000000-0000-0000-0000-{number:012d}") for number in (21, 22))
    store = JobStore(tmp_path / "jobs", uuid_factory=lambda: next(ids))
    child = store.create(JobRequest(source="queued.wav"))
    batch = store.create_batch(tmp_path, (child.id,))
    progress = batch.progress.advance(
        stage=JobStage.BATCH_PROCESSING, percent=0, message="processing"
    )
    batch = batch.model_copy(
        update={"status": JobStage.BATCH_PROCESSING, "progress": progress}
    )
    store.save_batch(batch)
    progress = batch.progress.advance(
        stage=JobStage.BATCH_PROCESSING, percent=100, message="incorrect failure"
    )
    batch = batch.model_copy(
        update={
            "child_outcomes": {child.id: JobStage.FAILED},
            "failure_count": 1,
            "progress": progress,
        }
    )

    with pytest.raises(InvalidJobTransition, match="does not match child job"):
        store.save_batch(batch)

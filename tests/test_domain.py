from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from engine.domain import (
    AudioStream,
    CategorySignal,
    ContentContext,
    JobRequest,
    JobRecord,
    JobStage,
    SignalSet,
    TimeRange,
    TimelineEntry,
)
from engine.events import Progress


def test_context_is_not_container_boolean():
    assert ContentContext.VIDEO.value == 1
    assert ContentContext.MUSIC.value == 3
    assert ContentContext.OTHER.value == 6
    assert ContentContext.UNKNOWN.value == 7


@pytest.mark.parametrize("start,end", [(-0.1, 1), (2, 1)])
def test_time_range_rejects_negative_or_reversed_timestamps(start: float, end: float):
    with pytest.raises(ValidationError):
        TimeRange(start_seconds=start, end_seconds=end)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["start_seconds", "end_seconds"])
def test_time_range_rejects_non_finite_timestamps(field: str, value: float):
    values = {"start_seconds": 0.0, "end_seconds": 1.0, field: value}
    with pytest.raises(ValidationError):
        TimeRange(**values)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_other_durable_domain_floats_reject_non_finite_values(value: float):
    from engine.domain import AcousticSignals, SourceMetadata

    with pytest.raises(ValidationError):
        SourceMetadata(duration_seconds=value)
    with pytest.raises(ValidationError):
        AcousticSignals(confidence=value)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_category_confidence_is_bounded(confidence: float):
    with pytest.raises(ValidationError):
        CategorySignal(taxonomy_id="80DV8O", confidence=confidence)


def test_category_signal_accepts_official_alphanumeric_taxonomy_ids():
    signal = CategorySignal(taxonomy_id="80DV8O", confidence=0.75)
    assert signal.taxonomy_id == "80DV8O"


@pytest.mark.parametrize("taxonomy_id", ["", "  ", "unsafe/id", "line\nbreak"])
def test_category_signal_rejects_blank_or_unsafe_taxonomy_ids(taxonomy_id: str):
    with pytest.raises(ValidationError):
        CategorySignal(taxonomy_id=taxonomy_id, confidence=0.75)


def test_prepared_media_rejects_duplicate_audio_stream_indexes():
    with pytest.raises(ValidationError):
        from engine.domain import PreparedMedia

        PreparedMedia(
            source_path="source.mp4",
            audio_path="audio.wav",
            container_kind="video",
            content_context_hint=ContentContext.VIDEO,
            selected_audio_stream=0,
            available_audio_streams=(
                AudioStream(index=0, codec_name="aac"),
                AudioStream(index=0, codec_name="aac"),
            ),
            duration_seconds=12,
        )


def test_ortb_content_is_allowed_only_for_complete_jobs():
    with pytest.raises(ValidationError):
        JobRecord(id="job-1", status=JobStage.TRANSCRIBING, ortb_content={"id": "x"})

    complete = JobRecord(id="job-1", status=JobStage.COMPLETE, ortb_content={"id": "x"})
    assert complete.ortb_content == {"id": "x"}


def test_job_and_batch_records_start_with_serializable_queued_progress():
    from engine.domain import BatchRecord

    job = JobRecord(id="job-1")
    batch = BatchRecord(id="batch-1", source_directory="media")

    assert job.progress.stage is JobStage.QUEUED
    assert job.progress.revision == 0
    assert batch.progress.stage is JobStage.QUEUED
    assert JobRecord.model_validate_json(job.model_dump_json()).progress.revision == 0
    assert BatchRecord.model_validate_json(batch.model_dump_json()).progress == batch.progress


def test_record_json_values_are_deeply_immutable_and_round_trip():
    from engine.domain import ModelProvenance

    job = JobRecord(
        id="job-1",
        status=JobStage.COMPLETE,
        ortb_content={"data": [{"segment": {"id": "sentiment"}}]},
        provenance=ModelProvenance(
            app_version="0.1",
            package_versions={"pydantic": "2"},
            model_revisions={"asr": "abc"},
        ),
    )

    with pytest.raises(TypeError):
        job.ortb_content["id"] = "new"  # type: ignore[index]
    with pytest.raises(TypeError):
        job.ortb_content["data"][0]["segment"]["id"] = "new"  # type: ignore[index]
    with pytest.raises(TypeError):
        job.provenance.package_versions["pydantic"] = "new"  # type: ignore[index]
    with pytest.raises(TypeError):
        job.provenance.model_revisions["asr"] = "new"  # type: ignore[index]

    dumped = job.model_dump()
    restored = JobRecord.model_validate_json(job.model_dump_json())
    assert restored.model_dump() == dumped


def test_omitted_progress_derives_from_complete_status_and_round_trips():
    from engine.domain import BatchRecord

    job = JobRecord(id="job-1", status=JobStage.COMPLETE)
    batch = BatchRecord(id="batch-1", status=JobStage.COMPLETE, source_directory="media")

    for record, record_type in ((job, JobRecord), (batch, BatchRecord)):
        assert record.progress.stage is JobStage.COMPLETE
        assert record.progress.percent == 100
        assert record_type.model_validate_json(record.model_dump_json()) == record


def test_records_reject_supplied_status_and_progress_stage_mismatches():
    from engine.domain import BatchRecord

    progress = Progress(stage=JobStage.TRANSCRIBING, percent=50, message="window 1")
    with pytest.raises(ValidationError):
        JobRecord(id="job-1", status=JobStage.ANALYSING, progress=progress)
    with pytest.raises(ValidationError):
        BatchRecord(
            id="batch-1",
            status=JobStage.ANALYSING,
            source_directory="media",
            progress=progress,
        )


@pytest.mark.parametrize(
    ("record_type", "kwargs"),
    [
        (JobRecord, {"id": "job-1"}),
        ("batch", {"id": "batch-1", "source_directory": "media"}),
    ],
)
def test_invalid_status_preserves_pydantic_validation_error(record_type, kwargs):
    from engine.domain import BatchRecord

    if record_type == "batch":
        record_type = BatchRecord
    with pytest.raises(ValidationError, match="status"):
        record_type(status="bogus", **kwargs)


def test_valid_timeline_job_json_round_trips():
    job = JobRecord(
        id="job-1",
        status=JobStage.COMPLETE,
        timeline=(
            TimelineEntry(
                time_range=TimeRange(start_seconds=0, end_seconds=12.5),
                signals=SignalSet(),
            ),
        ),
    )

    assert JobRecord.model_validate_json(job.model_dump_json()) == job


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_ortb_content_rejects_non_finite_values_at_any_depth(value: float):
    with pytest.raises(ValidationError):
        JobRecord(
            id="job-1",
            status=JobStage.COMPLETE,
            ortb_content={"data": [{"segment": {"confidence": value}}]},
        )


@pytest.mark.parametrize("payload", [{"value": object()}, {1: "not-a-string-key"}])
def test_ortb_content_rejects_unsupported_json_values_and_keys(payload: object):
    with pytest.raises(ValidationError):
        JobRecord(id="job-1", status=JobStage.COMPLETE, ortb_content=payload)


@pytest.mark.parametrize("index", [True, False, 1.0])
def test_job_request_rejects_ambiguous_audio_stream_index(index: object):
    with pytest.raises(ValidationError, match="audio_stream_index"):
        JobRequest(source="input.mp3", audio_stream_index=index)


def test_job_request_accepts_and_serializes_integer_audio_stream_index():
    request = JobRequest(source="input.mp3", audio_stream_index=2)

    assert JobRequest.model_validate_json(request.model_dump_json()).audio_stream_index == 2

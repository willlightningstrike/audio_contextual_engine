from engine.domain import ContentContext, JobRecord, JobRequest
from engine.lifecycle import JobStage
from engine.standards import TaxonomyRow
from scripts.run_pipeline_probe import (
    build_measurements,
    persist_summary,
    summarize_record,
)


def test_summary_reports_predicted_taxonomy_labels_for_completed_job():
    record = JobRecord(
        id="probe-job",
        status=JobStage.COMPLETE,
        request=JobRequest(source="/tmp/camera.wav"),
        language="en",
        resolved_content_context=ContentContext.VIDEO,
        ortb_content={"cat": ["633", "255"]},
    )
    taxonomy = (
        TaxonomyRow("633", "Cameras", 1, None, ("Technology", "Cameras")),
        TaxonomyRow("255", "Photography", 1, None, ("Hobbies", "Photography")),
    )

    assert summarize_record(record, taxonomy) == {
        "job_id": "probe-job",
        "status": "complete",
        "source": "/tmp/camera.wav",
        "progress": 100.0,
        "message": "complete",
        "language": "en",
        "context": 1,
        "categories": [
            {"id": "633", "name": "Cameras"},
            {"id": "255", "name": "Photography"},
        ],
        "error": None,
    }


def test_summary_keeps_failed_job_error_without_claiming_categories():
    record = JobRecord(id="probe-job", status=JobStage.FAILED, error="RuntimeError")

    assert summarize_record(record, ()) == {
        "job_id": "probe-job",
        "status": "failed",
        "source": None,
        "progress": 0.0,
        "message": "failed",
        "language": None,
        "context": None,
        "categories": [],
        "error": "RuntimeError",
    }


def test_measurements_report_peak_process_and_mlx_memory_for_macos():
    assert build_measurements(
        started_at=10.0,
        stopped_at=12.5,
        max_rss=4_000_000,
        platform_name="darwin",
        mlx_memory={"active_memory_bytes": 1, "peak_memory_bytes": 3},
        cancellation_requested_at=None,
    ) == {
        "wall_seconds": 2.5,
        "peak_process_rss_bytes": 4_000_000,
        "mlx_memory": {"active_memory_bytes": 1, "peak_memory_bytes": 3},
        "cancellation_request_to_stop_seconds": None,
    }


def test_measurements_report_cancel_request_to_worker_stop_latency():
    result = build_measurements(
        started_at=10.0,
        stopped_at=14.0,
        max_rss=4096,
        platform_name="linux",
        mlx_memory=None,
        cancellation_requested_at=12.25,
    )

    assert result["wall_seconds"] == 4.0
    assert result["peak_process_rss_bytes"] == 4096 * 1024
    assert result["mlx_memory"] is None
    assert result["cancellation_request_to_stop_seconds"] == 1.75


def test_persist_summary_keeps_a_separate_report_for_each_job(tmp_path):
    job_id = "61af3c6d-97ce-4d5d-b924-2a1d4a0067bd"
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)
    summary = {"job_id": job_id, "status": "complete", "categories": []}

    report_path = persist_summary(jobs_dir, summary)

    assert report_path == job_dir / "probe-report.json"
    assert report_path.read_text() == (
        '{\n  "job_id": "61af3c6d-97ce-4d5d-b924-2a1d4a0067bd",\n'
        '  "status": "complete",\n  "categories": []\n}\n'
    )

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_runner():
    path = Path(__file__).parents[1] / "scripts/verify_text_classification.py"
    spec = importlib.util.spec_from_file_location("verify_text_classification_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing_result() -> dict:
    return {
        "status": "passed",
        "cases": [{"error_code": None, "payload": {"primary_category": "633"}}],
        "cancellation": {"passed": True},
        "memory": {
            "peak_process_rss_bytes": 10,
            "process_rss_ceiling_bytes": 20,
        },
        "registry_release": True,
    }


def test_required_gate_rejects_a_failed_cancellation_probe():
    runner = _load_runner()
    result = _passing_result()
    result["cancellation"] = {"passed": False}

    verdict = runner.required_gate_verdict(result, require_cancellation=True)

    assert verdict == {
        "passed": False,
        "failures": ["CANCELLATION_TARGET_FAILED"],
    }


def test_required_gate_rejects_peak_rss_above_the_process_ceiling():
    runner = _load_runner()
    result = _passing_result()
    result["memory"]["peak_process_rss_bytes"] = 21

    verdict = runner.required_gate_verdict(result, require_cancellation=True)

    assert verdict == {
        "passed": False,
        "failures": ["PROCESS_RSS_CEILING_EXCEEDED"],
    }


def test_required_gate_rejects_failed_classification():
    runner = _load_runner()
    result = _passing_result()
    result["status"] = "failed"

    verdict = runner.required_gate_verdict(result, require_cancellation=True)

    assert verdict == {
        "passed": False,
        "failures": ["CLASSIFICATION_FAILED"],
    }


def test_required_gate_accepts_all_required_evidence():
    runner = _load_runner()

    verdict = runner.required_gate_verdict(
        _passing_result(), require_cancellation=True
    )

    assert verdict == {"passed": True, "failures": []}


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"status": "failed", "required_gate": {"passed": False}}, 1),
        ({"status": "passed", "required_gate": {"passed": True}}, 0),
    ],
)
def test_main_exit_code_reflects_classification_gate(
    monkeypatch,
    tmp_path: Path,
    result: dict,
    expected: int,
):
    runner = _load_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_text_classification.py", "classify", "--evidence-dir", str(tmp_path)],
    )
    monkeypatch.setattr(runner, "run_classification", lambda *_args, **_kwargs: result)

    assert runner.main() == expected

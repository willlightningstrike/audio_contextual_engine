# Clean-checkout reproducibility repair — 2026-10-09

Scope: restore three existing immutable benchmark manifests to version control and isolate fake-generator verifier tests from local model metadata. No production code, dependencies, live asset pins, or frozen benchmark results changed. Implementation base: `4ca875199eabb5749b3ccfdf4366e149e4fa67db`.

## RED: tracked-only baseline

From the development worktree:

```sh
repair_red_dir=$(mktemp -d /private/tmp/contextual-engine-repair-red.XXXXXX)
git archive fb7b156 | tar -x -C "$repair_red_dir"
```

The resulting directory was `/private/tmp/contextual-engine-repair-red.CbOeAt`. From that directory, using the existing dependency environment:

```sh
<project-root>/.worktrees/contextual-engine-phase1/.venv/bin/python -m pytest tests/test_verify_topic_support.py tests/test_english_acceptance.py -q
```

Result: **7 failed, 99 passed in 1.61s**, exit 1. Failures:

- `test_rows_and_durable_attempts`
- `test_probe_hashes_consumed_snapshots_when_factory_mutates_sources`
- `test_cancellation_finalizes_all_declared_windows_without_more_inference[True]`
- `test_cancellation_finalizes_all_declared_windows_without_more_inference[False]`
- `test_tv_podcast_supplement_is_accepted_only_at_its_frozen_digest`
- `test_tv_followup_is_accepted_only_at_its_frozen_digest`
- `test_corrective_supplement_is_accepted_only_at_its_frozen_digest`

The three acceptance failures were `FileNotFoundError` for the three restored manifests. The actual durable-attempt report recorded `MODEL_PREFLIGHT_ERROR` and this metadata error:

```text
FileNotFoundError: [Errno 2] No such file or directory: '/private/tmp/contextual-engine-repair-red.CbOeAt/data/models/qwen25-manifest.json'
```

## Test repair and development GREEN

Added the module-local autouse `model_metadata` fixture using the approved `test-only/fake-generator` repository and `test-only-revision` revision. The missing-metadata regression overrides `TEXT_MANIFEST`, checks the persisted report and empty journal, and verifies the factory was never called. The existing factory-failure regression was strengthened by replacing the manifest inside the failing factory, proving the returned identity and SHA256 reflect the bytes already read. Both regressions passed before fixture installation: `2 passed, 13 deselected in 0.39s`; they characterize already-correct production behavior rather than a new production fix.

The existing citation, consumed-snapshot hashing, durable-attempt, and cancellation assertions remain intact. `run_probe` and all production files remain unchanged.

From `<project-root>/.worktrees/contextual-engine-phase1`:

```sh
.venv/bin/python -m pytest tests/test_verify_topic_support.py tests/test_english_acceptance.py -q
.venv/bin/python -m pytest -q
```

Results: focused **107 passed in 1.14s**; dirty-development full suite **961 passed, 2 warnings in 16.57s**, both exit 0. The development count includes unrelated tracked edits and untracked tests; it is not the committed-only reproducibility count. Warnings were the installed FastAPI/Starlette `httpx` deprecation and `BlankAsrSegmentWarning` from the unrelated ASR recovery test. No dependency changes were made.

Raw SHA256 values, verified before and after the test repair:

| Manifest | Raw SHA256 |
| --- | --- |
| `acceptance-english-tv-podcast-v1.json` | `4d9c4e979fab21ab22942d8a20d8fde9e4af6474a9c6371d1b66fb1aed7be448` |
| `acceptance-english-tv-followup-v1.json` | `33057efc08af7d6756e8ed9a0eecfca1e7f78fe3e7a71793bef84e35cc7a6353` |
| `acceptance-english-corrective-v1.json` | `2768efa87fcdb7c9b91186c5cd0c2117e04f203d50741e3820cdb2467513a109` |

The acceptance tests independently validate canonical protocol digests. The raw hashes above do not replace those checks.

## Committed-only verification

Implementation commit: `e8c9e0aa2f0d03645c96a0fe6da05d554827e51b`, containing only the five task-owned files. Export command:

```sh
repair_green_dir=$(mktemp -d /private/tmp/contextual-engine-repair-green.XXXXXX)
git archive e8c9e0a | tar -x -C "$repair_green_dir"
```

Export cwd: `/private/tmp/contextual-engine-repair-green.dWqz1R`. From that cwd:

```sh
<project-root>/.worktrees/contextual-engine-phase1/.venv/bin/python -m pytest -q
```

Result: **934 passed, 1 warning in 17.80s**, exit 0; no failures. The sole warning was the installed FastAPI/Starlette `httpx` deprecation. This committed-only count is distinct from the dirty-development count of 961.

An existing-interpreter `python -c` check imported these modules, printed their resolved `__file__` paths, and asserted that each path was relative to the resolved export cwd:

| Module | Resolved location |
| --- | --- |
| `engine` | `/private/tmp/contextual-engine-repair-green.dWqz1R/engine/__init__.py` |
| `engine.models.asr` | `/private/tmp/contextual-engine-repair-green.dWqz1R/engine/models/asr.py` |
| `engine.signals.topic_verify` | `/private/tmp/contextual-engine-repair-green.dWqz1R/engine/signals/topic_verify.py` |
| `scripts.verify_topic_support` | `/private/tmp/contextual-engine-repair-green.dWqz1R/scripts/verify_topic_support.py` |

The same check asserted and printed that `data/models` and `benchmarks/runs` did not exist, and recursively checked that no `*.safetensors` existed in the export. Thus the test run used project source from the committed export and dependencies from the existing environment, without development assets.

Working and staged whitespace checks passed, the staged repair contained exactly the three manifests, amended test module, and this note, and all three raw manifest hashes remained unchanged. Unrelated development changes and untracked files were preserved. These results are appended in an evidence-only commit; independent review and local-main integration remain controller responsibilities.

## Review fix round 1: invalid metadata

Review Focus 3 identified missing direct regression coverage for invalid metadata. Added a parameterized regression overriding the module fixture's `TEXT_MANIFEST` with malformed JSON, a missing `model` object, a missing `repository`, and a missing `revision`. Each case invokes real `run_probe` with two declared cases, verifies the persisted report equals the returned report, checks absent model identity and the expected preflight exception type, checks every case fails with `MODEL_PREFLIGHT_ERROR` and no verified IDs or attempts, verifies an empty attempt journal, and asserts the generator factory was never called. Production code remains unchanged; these tests characterize existing fail-closed behavior.

From the same development worktree:

```sh
.venv/bin/python -m pytest tests/test_verify_topic_support.py tests/test_english_acceptance.py -q
```

Result: **111 passed in 0.82s**, exit 0, no test warnings. The prior 107/961/934 counts above describe the earlier implementation revision. The controller will perform the final committed-only full-suite verification after review; this round did not repeat the dirty-development full suite.

## Final review and local integration

After the invalid-metadata regressions, the controller exported committed tree `4a30bd64233ad83d23b914d64e47dc295c56bc24` with `git archive` into `/private/tmp/contextual-engine-repair-final.X4BHwQ`. From that export, using the same absolute interpreter and `-m pytest -q`, the suite passed **938 tests, 1 existing Starlette/httpx warning in 16.23s** (exit 0). Imports of `engine` and `scripts.verify_topic_support` resolved inside the export; `data/models`, `benchmarks/runs.local`, and recursive `*.safetensors` were absent. The controller independently recomputed all three approved raw manifest hashes, with exact matches.

Task review, scoped fix re-review, and final whole-repair review were clean after the invalid-metadata coverage fix. The final review found no Critical, Important, or new Minor issue. The existing dependency deprecation remains disclosed and outside this repair.

Local `main` fast-forwarded from `fb7b156` to `4a30bd6`. From `<project-root>`, `.worktrees/contextual-engine-phase1/.venv/bin/python -m pytest -q` passed **938 tests, 1 existing Starlette/httpx warning in 16.37s**, exit 0. No remote is configured and nothing was pushed. The development branch/worktree and unrelated edits, models, media, and evidence remain preserved. This final documentation closure does not change tested code or data.

## Scope limitation

This repair establishes test/data reproducibility with the existing interpreter and dependencies. It does not admit an acoustic model, resolve historical ASR failures, certify runtime/verifier integration, or establish production readiness. No model downloads, network calls, or real-model inference were performed.

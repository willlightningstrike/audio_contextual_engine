# Task 13: Integrated English Pipeline Verification

**Date:** September 26, 2026
**Host:** Apple M1 Max, 64 GB unified memory
**Scope:** Local real-model end-to-end probes for the integrated audio-only
pipeline. These are development observations, not Task 18 release acceptance.

## Vocal English results

All five clips completed with English transcripts and a validated result:

| Clip | Primary | Optional secondaries | Runtime | Peak process RSS | Peak MLX memory |
|---|---|---|---:|---:|---:|
| T02.wav | Road Trips | — | 47.19 s | 15.48 GB | 19.63 GB |
| B04.wav | Computer Peripherals | — | 64.30 s | 18.70 GB | 19.72 GB |
| Bon Jovi — *It's My Life* | Contemporary Hits/Pop/Top 40 | Single Life | 121.39 s | 19.41 GB | 19.73 GB |
| Adele — *Someone Like You* | Dating | Marriage and Civil Unions; Single Life | 126.61 s | 17.73 GB | 19.74 GB |
| Whitney Houston — *I Will Always Love You* | Adult Contemporary Music | Prom | 114.43 s | 15.29 GB | 19.69 GB |

T02's Road Trips result and Whitney's Prom secondary are weak/questionable
semantic predictions. The table preserves what the model returned; these are
not human-adjudicated successes. Adele's categories are plausible against the
relationship lyrics, but do not establish broad category accuracy.

The two full-length song probes each completed despite exactly one terminal
`VOCAL_PRESENCE_UNCERTAIN` range. Adele excluded 270.00–284.978503 seconds;
Whitney excluded 270.00–274.041905 seconds. No transcript or category was
assigned to either excluded tail. Detailed results are in the
[terminal uncertainty probe note](2026-09-26-terminal-uncertain-tail-completion.md).

## Negative routing and cancellation

| Clip | Expected class | Outcome | Diagnostic | Export |
|---|---|---|---|---|
| T04.wav | Instrumental music, no vocals | Failed; no categories | `VOCAL_PRESENCE_UNCERTAIN` | None |
| Y09.wav | Other sounds, no vocals | Failed; no categories | `VOCAL_PRESENCE_UNCERTAIN` | None |
| Y11.wav | Silence | Failed; no categories | `VOCAL_PRESENCE_UNCERTAIN` | None |
| B05.wav | Non-English | Failed; no categories | `NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE` (`zh`) | None |
| B04.wav, cancellation probe | English speech/media | Cancelled | Cancellation request to stop: 0.572 s | None |

The four failed/cancelled job directories have no `result.json`. The uncertain
vocal-screen results for T04/Y09/Y11 are deliberately reported as uncertainty,
not as confident no-vocal detections; user-provided listening adjudications
identify their content class.

## Deterministic verification

- `.venv/bin/python -m pytest tests/test_pipeline.py tests/test_pipeline_probe.py -q` — **20 passed**.
- `UV_OFFLINE=1 UV_CACHE_DIR=.uv-cache .venv/bin/python -m pytest -q` — **539 passed**.
- `scripts/audit_imports.py` — passed.
- `scripts/audit_dependencies.py` — passed.
- `git diff --check` — passed.

## Evidence limits

Raw probe reports and window journals are retained locally under the ignored
`benchmarks/runs.local/task13-terminal-probe/` directory. Job IDs:

- T02 `7096a71e-24a0-4c1f-bb52-1c48ee225cbe`
- B04 `a71e6801-b59a-4468-8503-0263407a06cd`
- Bon Jovi `61f1cc3e-e68e-4f36-96d0-e790c2c26400`
- Adele `bc2f8132-b52f-4acb-9552-29704d330e63`
- Whitney Houston `899d31d5-9062-4980-9c73-08071a30a9ad`
- T04 `76cdae39-b68e-4e01-a675-b4c3f77752c2`
- Y09 `9773b4f5-64aa-4855-955b-6627a2c861b0`
- Y11 `a18d0044-d43e-463e-ac62-2efb1390014c`
- B05 `88589ae5-cccc-49bf-badf-dd77d2d1ec3d`
- cancelled B04 `e1a0446d-b81b-42a9-9df4-d579630bc132`

These probes establish that the integrated pipeline, terminal-tail exception,
fail-closed negative routes and one real cancellation run work on the evidence
host. They do not establish general semantic accuracy, full-app resource
acceptance, or release readiness; those belong to Task 18.

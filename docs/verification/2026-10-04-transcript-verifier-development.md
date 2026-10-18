# Saved-transcript verifier development controls

Gate: **pass** for the four named, non-blind development controls. This is an isolated verifier replay of saved v1 transcripts and proposals, not new whole-job success, blind accuracy evidence, runtime integration, or a Task 18 release decision. No audio was retranscribed, model downloaded, or frozen v1 evidence rewritten.

The actual Metal run on 2026-10-04 exited 0. The immutable report is `benchmarks/runs.local/english-transcript-verifier-v1/20261004T184611Z.json`, SHA256 `55419b0c9f04e5902299d0fa10fab5d6f3b8abb6526ca9d5ef4f2ffc9a651a1b`. Its prediction-first sidecar is `benchmarks/runs.local/english-transcript-verifier-v1/20261004T184611Z.attempts.jsonl`, SHA256 `ca1e3982257b6c1fb5d6a8d4b920f64f5085941ca5b1df6c41d07eb2bc76a075`. The sidecar preserves all 12 raw attempts and prompt hashes before their parsing; the report retains parsed judgments, attempt errors, source hashes, exact job paths and partial ASR history. Both destinations were opened exclusively before inference.

Policy `english-transcript-verifier-v1` used the installed, inventory-verified `mlx-community/Qwen2.5-32B-Instruct-4bit`, revision `2938092373e5f97b95538884112085364c2da315`. The manifest SHA256 is `70769ae1399cfa9dab1e5e43a866c59afbdc18eaea2d00cdff302d9f9fdd4efe`. The generator uses the existing constrained MLX adapter and the verifier's unchanged 128-token cap, five-call maximum and bounded alternatives. Case names and media filenames are never prompt evidence.

All input job directories are under `benchmarks/runs.local/english-topic-sufficiency-v1/terminal-run-20260928-01-artifacts-20260929T004457Z/`. The exact job UUIDs and both journal hashes are stored in the report. The original `terminal-run-20260928-01.json` still hashes to `bca0eedfd36400631fc2cb4e2a19a5963f403a0017cf0014429a4fe06c504092`.

The following ranges are job-local seconds, matching exact saved analysis windows. Citation indices are local to each analysis window. Every listed attempt finished with `stop`; no preflight, generation, schema or alignment error occurred.

| Case | Window | Saved proposed IDs | Verifier judgments and citation indices | Verified IDs |
| --- | --- | --- | --- | --- |
| `sony-camera-speech-30-90` | 0–59.36 | 514 | 514 unsupported; 633 supported [0,2,4,5,6,7,8,9,10] | 633 |
| `backstreet-vocal-90-120` | 9.459999999999996–25.64 | 363 | 574 unsupported; 225 unsupported; 505 unsupported | none |
| `hario-coffee-speech-120-300` | 0–89.8 | 222 | 222 supported [0,2,3,4,5,6] | 222 |
| `hario-coffee-speech-120-300` | 90–179.82 | 222 | 222 supported [0,1,10,11,12,13,14,16,17,18,19,20] | 222 |
| `whitney-houston-full-audio` | 1.58–88.74 | 186 | 186 supported [0,1,2,3] | 186 |
| `whitney-houston-full-audio` | 92–179.74 | 186 | 186 supported [1,6,7,8] | 186 |
| `whitney-houston-full-audio` | 180–265.06 | 339 | 186 unsupported; 173 unsupported; 200 unsupported | none |

Sony rejects the Hunting and Shooting lexical mistake (514) and accepts Cameras and Camcorders (633). Backstreet makes zero Music-branch calls and emits no verified topic; its three calls evaluate non-Music alternatives only. Hario retains Non-Alcoholic Beverages (222) in both windows. Whitney's Family and Relationships topic (186) remains assessable and is supported in the first two windows. Its final lyric window abstains after rejecting the three eligible alternatives; the excluded saved Music proposal 339 is never evaluated as transcript topic evidence. The controls establish these bounded behaviors only. Structural citation validity does not independently prove semantic entailment.

Inherited ASR limitations remain visible. Whitney's 270–274.041938 terminal transcription unit has `VOCAL_PRESENCE_UNCERTAIN` at `vocal_screening`, so its reconstructed transcription status is `partial`; there is no verifier window or category for that failed unit. The other three saved runs reconstruct as `ready`, which describes error-free saved transcription units, not independently checked word accuracy. Sony's retained text includes “sprite monitoring” alongside “bright monitoring”; the probe does not correct this lexical fault. Lyric fragments, omissions and timestamps were reused unchanged and were not rescored against audio. In particular, these observations do not turn the old Whitney whole-file job or frozen acceptance results into a new success.

The isolated process took 156.364800084 seconds including model verification/loading and replay. Peak process RSS was 10,373,201,920 bytes; MLX allocator peak was 19,787,894,824 bytes, with final active 18,431,289,352 and cache 272,114,938 bytes. RSS and MLX counters overlap and must not be added. Both observed peaks are below the host's 64 GiB; no system swap observation was made, and this is not a general resource acceptance claim.

Validation: first targeted execution failed during collection because `scripts.verify_topic_support` did not exist. The initial focused reader/verifier suite then passed 63 tests. After duplicate-record, partial-ASR and strict saved-success JSON regressions, the probe suite passed 10 tests and the complete suite passed **820 tests**, with the same existing Starlette deprecation and blank-ASR warnings. The CLI result above is an actual fresh inference run, separate from those unit checks. Runtime integration and independent acoustic genre feasibility remain later work.

Post-run code review corrections (2026-10-04) finalize explicit cancellation reports, bind input parsing and hashes to the same captured journal bytes, preserve valid model identity on loading failure, and normalize oversized-integer parser errors. Focused verifier tests passed 71; the then-current complete suite passed 860, including the independent acoustic-ledger tests, with the same two baseline warnings. These bookkeeping/error-boundary changes did not change the verifier prompt, policy, limits, saved transcripts, frozen v1 evidence, or the immutable live-control report, and no repeated inference is claimed.

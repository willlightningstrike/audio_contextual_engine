# Transcript-only v2 end-to-end evaluation

**Status:** complete; limited experimental evidence, not a production-readiness claim.

**Run:** `transcript-only-v2-evaluation-20261010` on the 64 GiB Apple Silicon host.

**Protocol:** [`acceptance-english-transcript-only-v2.json`](../../benchmarks/acceptance-english-transcript-only-v2.json), `english-audio-transcript-only-v2`.

## Bottom line

The full frozen set was attempted: all 308 available cases reached a terminal result. The engine published a verified primary label for 96 cases. It abstained for insufficient topic evidence in 99; 61 were gated as non-English or language-unconfirmed; 35 had no confident vocal-presence decision; and 17 had explicit generation/classification/model failures. Those five dispositions sum to the 308 available cases. Fifty-one other inventory inputs were unavailable and were not run.

This supports trying the build as an explicitly experimental, human-reviewed prototype. It does not support unattended contextual labeling: abstention is frequent, a few verified labels are visibly wrong, and structured-output failures remain. No population accuracy was calculated. This is a non-blind, mixed-provenance development set, and some cases are excerpts, aliases, or near duplicates.

## Frozen scope and setup

- Audio only, including audio tracks extracted from video. **No video frames were examined.**
- English content is the classification scope. Non-English/unconfirmed language, uncertain vocals, instrumental-only audio, and other ineligible audio receive no contextual label.
- YAMNet is used only to screen speech/singing eligibility. No foreground/genre classifier was enabled, and no IAB music-genre label was emitted. Transcript-topic support requires primary-role citations covering more than 50% of usable transcript duration and at least 25% of selected-audio duration; primary plus at most two optional secondary categories. Documentation correction on 2026-10-11: the earlier 70% statement described the unenabled acoustic foreground/genre policy, not the transcript code used in this run. Raw results and policy code are unchanged.
- Apple M1 Max, arm64, macOS 27.0.1, 68,719,476,736 bytes physical memory (64 GiB). Metal and the offline setup gate passed.
- ASR: `mlx-community/whisper-large-v3-turbo-asr-fp16` revision `624c19c9af5603fa73b83bce14d4aeea96156d18`; embeddings: `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` revision `6c3ae70858513f1a78e9cdca3cae330d9075cd2a`; text model: `mlx-community/Qwen2.5-32B-Instruct-4bit` revision `2938092373e5f97b95538884112085364c2da315`; YAMNet was converted with `yamnet-tf-reference-to-mlx-npz-v1`. Existing model weights were referenced in place; none were downloaded or copied for this run.
- Frozen manifest canonical SHA-256: `c168581ecbe34b7d78f46448407c8c18431abb346b532478016e0e85c9836a2c`; byte SHA-256: `9a8d1e12fd39a74caff4dda75d09cc468577ffbfb72d447d01a4b2c28dcf8dac`. The freeze record binds the same canonical digest. The 359 inventory IDs comprise 308 available cases and 51 unavailable inputs; 15 source-mapping claims remain explicitly unverified.

## Dispositions and labels

The runner’s raw `outcome` field has a precedence defect: it checks `status == failed` before the explicit abstention code, so it calls all 99 `INSUFFICIENT_TOPIC_EVIDENCE` cases technical failures. The raw records were not changed. The linked per-clip CSV preserves the raw status/outcome/error and adds a separate normalized disposition using the final category, explicit abstention code, range-level technical codes, language gate, and vocal-screen evidence. The 113 generic `ValueError` terminal records normalize to 61 language-gated cases, 35 vocal-uncertain cases, and 17 true technical failures; none is silently counted as a successful label.

| Normalized disposition | Cases | Meaning |
| --- | ---: | --- |
| Classified | 96 | A final verified primary category was emitted; 6 cases also have secondaries (7 secondary labels total). |
| Semantic abstention | 99 | Explicit `INSUFFICIENT_TOPIC_EVIDENCE`; no final category. |
| Non-English / unconfirmed language | 61 | Out of Phase 1 scope; no final category. |
| Vocal presence uncertain | 35 | Eligibility could not be established confidently; no final category. |
| Technical failure | 17 | Explicit incomplete generation, invalid classification output, or model failure. |

The 17 technical cases contain 33 failed ranges: 20 `INVALID_CLASSIFICATION_OUTPUT`, 11 `INCOMPLETE_GENERATION`, and 2 `MODEL_FAILURE`. These counts are ranges, not independent cases. Across the run, 62 ranges were language-gated, 4 had `NO_USABLE_TRANSCRIPT`, and 103 ranges were excluded for `VOCAL_PRESENCE_UNCERTAIN`.

The 96 primary labels span 19 IAB categories. Most common were:

| Primary category | IAB ID | Cases |
| --- | ---: | ---: |
| Men’s Jewelry and Watches | 581 | 28 |
| Cameras and Camcorders | 633 | 21 |
| Video Production | 268 | 11 |
| Computer Peripherals | 601 | 11 |
| Non-Alcoholic Beverages | 222 | 7 |

The complete 308-row table includes every clip’s primary/secondary label and taxonomy path (when present), normalized and raw outcome, language, source extent, transcript/window coverage, failed/excluded/abstained ranges, exact verifier citation ranges, wall time, peak sampled RSS/MLX memory, cancellation latency, export hash, and pre-inference reference label when present:

- [Per-clip results CSV](../../benchmarks/runs.local/transcript-only-v2-evaluation-20261010/results-standalone/per-clip-results.csv)
- [Raw case records](../../benchmarks/runs.local/transcript-only-v2-evaluation-20261010/results-standalone/case-records.jsonl)
- [Run metadata](../../benchmarks/runs.local/transcript-only-v2-evaluation-20261010/results-standalone/run-metadata.json) and [completion record](../../benchmarks/runs.local/transcript-only-v2-evaluation-20261010/results-standalone/run-completion.json)

### Pre-inference reference checks

These five clips had confirmed reference labels frozen before inference. The comparison is descriptive only; it is not a blind-accuracy estimate.

| Clip | Frozen reference | Run result |
| --- | --- | --- |
| B01 (`acoustic-074`) | Cameras and Camcorders (633) primary; Photography (255) secondary | Cameras and Camcorders primary; Video Production (268) secondary. Primary agrees; secondary does not. |
| B02 (`acoustic-069`) | Non-Alcoholic Beverages (222) primary | Non-Alcoholic Beverages (222). |
| B03 (`acoustic-107`) | Men’s Jewelry and Watches (581) primary | Men’s Jewelry and Watches (581). |
| B04 (`acoustic-005`) | Computer Peripherals (601) primary | Semantic abstention; no final label. |
| B06 (`acoustic-119`) | Men’s Jewelry and Watches (581) primary | Men’s Jewelry and Watches (581). |

Two tiny, likely near-duplicate Tissot watch-review excerpts (`acoustic-103` and `acoustic-104`) produced the known false positive **Remodeling & Construction (276)** from the phrase “brushed and polished surfaces” / “steel frame.” They are not independent evidence of accuracy. The full Tissot PRX review (`supplied-full-043`) instead abstained, despite all four classified windows suggesting Men’s Jewelry and Watches; the stricter clip-level evidence policy did not promote it.

## Coverage, latency, and resources

- Scope mix: 60 full tracks (66,537.2 seconds / 18.48 hours), 80 exact excerpts (4,615 seconds), and 168 inputs with unknown original-media extent (5,949.0 seconds). Overall selected audio totals 77,101.28 seconds (21.42 hours). Unknown-extent results must not be read as whole-original-file results.
- 1,391 of 1,562 attempted windows yielded usable transcription; the run contains 31,213.6 seconds of transcript timestamps. These are coverage observations, not accuracy measures.
- Per-case wall times sum to 47,403 seconds (13.17 hours). For cases with a final label, median wall time was 67.1 seconds, p90 116.1 seconds, and maximum 990.1 seconds. The slowest case was a 93.3-minute baseball program: 6,526.2 seconds wall time, with no final clip label because several windows failed verification. It did process the full selected track and retained a strong partial Baseball signal.
- Maximum per-job sampled process RSS was 19,667,435,520 bytes (18.3 GiB); maximum sampled MLX memory was 20,386,367,714 bytes (19.0 GiB). Sampling was once per second. This is evidence from one 64 GiB M1 Max, not a support claim for smaller machines.
- No cancellation was requested in this evaluation; therefore it produced no live cancellation-latency observation. Task 3’s cancellation/progress checks remain separate from these case runs.

Longer media show both value and limitations. For example, the 20-minute MotionVFX freeze-frame tutorial transcribed 40/40 windows (1,074.8 seconds with transcript timing), but the final result abstained because window topics split across Video Production, Computer Animation, Video Software, 3-D Graphics, and Photo Editing Software. The full-track ledger preserves each window’s evidence and disposition.

## Reconnect acceptance and verification

Per the user-approved substitute, the successful service/API SSE reconnect check plus deterministic UI reconnect regression are accepted instead of a manual live-browser transport disconnect/reconnect. **The manual browser-level EventSource transport reconnect remains unverified.**

Final revalidation with `scripts/transcript_only_acceptance.py` passed against the recorded parent project source root: canonical manifest digest matched; 308 available inputs, 51 unavailable inputs, and 15 unverified source claims were confirmed. The run-completion record reports 308/308; all 308 progress and resource-sample files are retained, with 96 successful exports. The generated CSV has 308 unique case IDs and its normalized disposition counts match the raw case records. Offline dependency and import audits both passed.

Final deterministic suite: `.venv/bin/python -m pytest -q -m 'not real_models'` — **1,086 passed**, with two disclosed pre-existing warnings (Starlette/httpx deprecation and expected `BlankAsrSegmentWarning`).

**Recommendation:** keep this behind an experimental opt-in with human review. Prioritize fixing strict structured-output/citation failures and reducing semantic false positives before broadening use; maintain the conservative abstention policy until changes are separately validated. Keep audio-only scope, no genre output, and no video-frame inference for this Phase 1 result.

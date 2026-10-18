# Phase 1 English Acceptance Evidence

The separately versioned [English topic-sufficiency v1 evaluation](2026-09-27-english-topic-sufficiency-evaluation.md) froze six non-blind challenge inputs and a code-pinned digest. Its later interactive run failed the execution and coverage gates; qualitative review and release remain pending. The historical results below retain their original interpretation.

**Date:** 2026-09-27
**Status:** **Fail — the live run has an execution failure and incomplete negative-control coverage; semantic and resource gates remain pending.**
**Protocol:** `benchmarks/acceptance-english-v1.json`
**Protocol SHA-256:** `97820db9e1cf2de3a560bb3c19183fbc2c3041ea9d4dd1534ed1692f4065112d`

## Approved acceptance policy

The semantic review is qualitative and prediction-first. Each completed clip
must show the model's primary category, zero to two optional secondaries,
keywords, entities, sentiment, transcript coverage, and errors. The frozen
protocol defines no semantic accuracy percentage. Predictions are not
converted into ground truth; later user feedback belongs in a separate
post-result review record. The semantic gate can clear only after every case
has an explicit `approved` review; a `revision_requested` review fails that
gate. Neither outcome is converted into a numeric accuracy claim.

Phase 1 uses audio only, including the audio tracks of video files. No video
frames are read. English speech and intelligible English singing are in scope;
non-English, instrumental-without-vocals, other no-vocal sounds, and silence
are negative routing controls. Instrumental music without vocals is an
out-of-scope error, not a music-category success.

## Frozen inputs

All inputs reuse project or development media and are explicitly **not an
independent holdout set**. Exact source hashes, source ranges, normalized
audio-input hashes, and context overrides are in the protocol. The six video
clips were converted to mono 16 kHz PCM excerpts using their audio stream only.

| Clip | Frozen source range | Stratum |
|---|---:|---|
| B04 keyboard/peripherals speech | 00:00–01:30 | Short English speech |
| Sony A7Cii tips | 00:30–01:30 | English video speech |
| Hario Switch coffee brewing | 02:00–05:00 | Longer English speech |
| Final Cut Pro AI masking | 01:00–02:00 | English video speech |
| Cartier Tank review | 01:00–02:00 | English video speech |
| Backstreet Boys — “Everybody” | 01:00–02:00 | English vocal music |
| Adele — “Hello” | 01:00–02:00 | English vocal music |
| B05 | 00:00–01:30 | Non-English negative |
| T04 | 00:00–00:05 | Instrumental, no vocals |
| Y09 | 00:00–00:05 | Other sounds, no vocals |
| Y11 | 00:00–00:05 | Silence |

These are selected development examples, not blind semantic labels. This
selection has no TV-program or podcast example, so it cannot support a claim
about those formats. It also does not explicitly represent sparse-speech and
speech-plus-background-music strata required by the Task 18 plan. The frozen
set was not silently expanded after approval; those coverage items remain
open. The existing Task 13 model observations remain historical development
evidence and are not relabelled as Task 18 results.

## Initial headless attempt (historical)

- Protocol validation passed with zero protocol errors.
- All 11 source and analysis-input hashes and durations matched the frozen
  protocol before inference.
- Host memory reported **68,719,476,736 bytes (64 GiB)**.
- The one-element MLX device preflight failed with:
  `RuntimeError: [metal::load_device] No Metal device available`.
- No model weights loaded, no clip was attempted, and no model was downloaded.
- The 64 GiB Apple Silicon/macOS preflight, frozen protocol digest, both media
  hashes, taxonomy membership, explicit errors, qualitative approvals and
  cancellation limits are enforced by the harness.
- Cancellation evidence is timed at the observed child-process exit, before
  reader/resource cleanup. A Ctrl-C handled between child runs is recorded as
  an interruption but does not fabricate a successful batch process-stop
  measurement; that batch boundary remains unproven. Malformed saved
  attempt/review collections now produce a failed audit report rather than an
  uncaught iteration error.
- The acceptance CLI returned exit status 1 and recorded
  `release_decision: pending`. Gate states: protocol integrity **passed**;
  execution, coverage, semantic review, and resource evidence **pending**;
  routing, schema, and export integrity **not applicable** because no clip
  ran.
- Deterministic suite: `.venv/bin/python -m pytest -q` — **634 passed, 1
  warning**.
- `scripts/audit_imports.py` and `scripts/audit_dependencies.py` — **passed**.
- Existing Task 17 wheel smoke — **passed** (703 taxonomy rows, static assets
  present). The first attempt hit the restricted global uv cache; rerunning
  offline with the worktree-local `.uv-cache` succeeded.

The initial headless run record is
`benchmarks/runs.local/english-acceptance-v1/acceptance-report-metal-blocked-v7.json`.
The extracted audio inputs are under the ignored local directory
`benchmarks/runs.local/english-acceptance-v1-inputs/`.

## Interactive Terminal run and offline rescore

The 2026-09-27 interactive Terminal run attempted all 11 frozen inputs. The
original report is `benchmarks/runs.local/english-acceptance-v1/terminal-run-20260927-v3.json`.
Its export comparison incorrectly passed a frozen in-memory tuple to the JSON
validator, marking six completed jobs as failures. A regression test now covers
that comparison, and a second regression keeps saved window predictions visible
when a job has no final export. The saved jobs were rebuilt into
`benchmarks/runs.local/english-acceptance-v1/terminal-run-20260927-v3-rescored-v2.json`.
The rebuild did not rerun model inference. The original report and artifacts
remain unchanged; the revised report records the harness source SHA-256.

Completed English results below have **zero secondary categories**. The
duration is transcript evidence within the exact frozen source range, not a
claim that every second contains speech or lyrics:

- **B04 keyboard, 00:00–01:30:** 87.08/90.00 seconds. Primary Computer
  Peripherals (601); keywords `keyboards`, `meetups`, `switches`; entities
  `discord server`, `LTC Nimbleback`; sentiment neutral, 0.0; errors none.
- **Sony A7Cii, 00:30–01:30:** 56.90/60.00 seconds. Primary Cameras and
  Camcorders (633); keywords `bright monitoring`, `dark environment`,
  `hidden function`, `sprite monitoring`; entities `Custom Key Dial Set`,
  `Menu`, `Operation Customize`, `Setup`; sentiment neutral, 0.0; errors none.
- **Hario Switch, 02:00–05:00:** 165.90/180.00 seconds. Primary Non-Alcoholic
  Beverages (222); keywords `coffee`, `brewer`, `filter`, `brewing`,
  `El Salvador`; entities `glass brewer`, `Hario`, `11 grams`, `200 grams`,
  `93 degrees`, `Concepcion de Ataku`, `Lemos Bella Vista`, `Tabi`, `V60`;
  sentiment positive, 0.6; errors none.
- **Final Cut Pro masking, 01:00–02:00:** 50.16/60.00 seconds. Primary Photo
  Editing Software (604); keywords `AI Person Mask`, `free 7-day trial`,
  `inside-outside mask technique`; entities `AI Person Mask` and
  `inside-outside mask technique`; sentiment neutral, 0.0; errors none. This
  photo-editing label needs qualitative review for a video-editing clip.
- **Cartier Tank, 01:00–02:00:** 57.18/60.00 seconds. Primary Men's Jewelry
  and Watches (581); keywords `coussin`, `Mike Nouveau`, `Tank Centrae`,
  `Tank Louis`, `Tank Normale`; the same named person/product details appear
  as entities; sentiment positive, 0.6; errors none.
- **Adele “Hello”, 01:00–02:00:** 37.84/60.00 seconds. Primary Family and
  Relationships (186); keywords `heal`, `time`, `years`; entity `California`;
  sentiment neutral, 0.0; errors none.

Other attempted inputs:

- **Backstreet Boys “Everybody”, 01:00–02:00:** 16.18/60.00 seconds of
  transcript evidence. The saved window at source 01:39.46–01:55.64 predicts
  Rock Music (363), no secondaries, keywords `rock` and `body`, entity
  `rock your body`, and neutral sentiment. There is **no final clip category**.
  Replaying the saved windows and classification through `completion_errors`
  yields `transcription_partial` and
  `transcription_window_failed:VOCAL_PRESENCE_UNCERTAIN`: the opening 30-second
  window is uncertain, followed by usable English lyrics. The approved
  completion exception applies only to one uncertain *final* window, so the
  job correctly failed; its durable error field says `ValueError`.
- **B05 non-English, 00:00–01:30:** `zh` was detected in the first 30 seconds;
  the job exited as `rejected_non_english` with
  `NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE`, no transcript/category/other semantic
  fields. Its processed range is 30/90 seconds, so the frozen v1 coverage gate
  fails even though early non-English rejection is the intended pipeline path.
- **T04 instrumental, Y09 other sounds, Y11 silence, each 00:00–00:05:** each
  returned `VOCAL_PRESENCE_UNCERTAIN` and `rejected_unconfirmed_vocal`, with
  no transcript, category, keywords, entities, or sentiment. This is an
  explicit out-of-scope outcome, not a verified detector assertion that each
  clip has no vocals.

The rescored gates are: protocol integrity **passed**; execution **failed**
(Backstreet Boys); frozen-range coverage **failed** (B05); schema and export
integrity **passed** for completed results; semantic review and resources
**pending**. Routing accuracy is **N/A** because no prior routing references
were frozen. The host reported 68,719,476,736 bytes (64 GiB) physical memory;
the largest recorded process RSS was 16,991,780,864 bytes and the largest MLX
peak was 19,721,576,630 bytes. Recorded swap endpoints and samples were zero;
model downloads were zero. Cold/warm timing and all four required live
cancellation probes are absent. The cold/warm probe was skipped because the
original harness misidentified the first completed job as failed at its export
comparison; the corrected harness will attempt it on a future live run. The
input set is reused development material, not an independent holdout.

The completed Terminal command used this output path once:

```sh
cd "<project-root>/.worktrees/contextual-engine-phase1"
.venv/bin/python scripts/english_acceptance.py \
  --protocol benchmarks/acceptance-english-v1.json \
  --output benchmarks/runs.local/english-acceptance-v1/terminal-run-20260927-v3.json
```

Do not rerun into the same path: the harness deliberately refuses to overwrite
evidence. Any future run needs a new output name and must keep the v1 failures
visible. A post-result approval must reference the exact case's
`prediction_sha256`; category approval does not automatically approve its
keywords, entities or sentiment.

## Remaining acceptance work

Six later-supplied MP4s were evaluated as a separately pinned TV/podcast
supplement; see the 2026-09-27 TV/podcast note (not published).
That supplement is not a replacement for this frozen v1 run and has its own
execution and coverage failures.

Two additional user-supplied TV MP4s were checked in a second, separately
pinned TV audio follow-up (not published). One English excerpt
completed with conflict-related labels; the other produced no final label
because every selected window had uncertain vocal presence. Neither result
changes the v1 or six-case supplement outcomes.
An explicitly post-result exploratory range of the second file later yielded
English dialogue and a gift-related primary label; it is not a substitute
for the frozen failed-to-classify excerpt.

A separately frozen corrective supplement (not published)
later processed four post-result/development-derived audio cases. Its two TV
excerpts and narrower Backstreet vocal excerpt completed; a 30-second B05
control was rejected as non-English. The older timestamp failure was
reproduced and narrowly addressed in the ASR adapter, but the single live
corrective rerun does not establish a failure rate. The corrective report is
`release_decision: pending`; it neither repairs earlier frozen results nor
approves provisional semantic predictions.

Keep the approved completion policy for the uncertain opening of the original
Backstreet Boys excerpt. Do not promote its partial label into a final result
or replace the failed frozen case after seeing its result. The new B05 control
checks only the explicit first-window rejection; it does not resolve the old
90-second protocol's coverage failure.

The current results need explicit qualitative review. A cold/warm same-input
timing check was captured for one supplemental TV excerpt, but timed
cancellation at extraction, ASR, classification and batch boundaries still
needs a Metal-enabled run. Supplemental English TV/podcast format examples
now exist, while sparse-speech and speech-plus-background-music strata remain
unverified. Manual UI
upload, reconnect, cancellation and byte-identical copy/download checks also
remain. The current report is not a release or full-application acceptance.

A later topic-sufficiency design (internal, not published)
defines abstention from unsupported IAB topics and material whole-file evidence.
Its implementation plan
has been implemented. The real-model semantic run remains blocked or pending;
implementation and deterministic checks do not establish semantic acceptance.
None of the historical results or gate states above is reinterpreted under the
new policy.

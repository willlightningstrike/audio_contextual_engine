# Task 10: ASR reliability investigation

Latest update: the approved `word_timestamps=true` plus contiguous terminal
anomaly policy is now production behavior at the MLX ASR adapter boundary. The
committed adapter recovered all five previously failed 30-second windows
without regressing the four controls, while preserving the strict timestamp
validator and a one-attempt ASR budget. The fresh three-excerpt audio-only
pipeline then passed transcription, retrieval and the production two-stage
classification contract with the approved labels. ASR cancellation completed
in 3.27 seconds and classifier cancellation in 10.22 seconds, both below the
35-second target. The final production evidence and conclusion are recorded in
the last section.

Subsequent approved change: the narrow one-ULP precision correction is now
implemented in `f2ec562`; the final section records its tests and saved-output
verification. Shorter-window retries remain experimental, not integrated.

Status: Task 10 complete September 26, 2026. Production evidence and common
gates passed; the final scoped review approved the implementation with no
remaining findings. Fresh verification: 138 focused tests passed, 471
full-suite tests passed, import/dependency audits passed and `git diff --check`
passed. Tasks 11–18 remain paused.

Diagnostic implementation commit: `e3405e5`; replay script/test checkpoint:
`9133e57`; review fixes: `a789fc6`. Full suite: 445 passed on September 19; import and dependency audits
passed (dependency audit requires `UV_OFFLINE=1 UV_CACHE_DIR=.uv-cache` here).
New failure-stage/reason fields persist through the job store without exposing
arbitrary exception messages. Cancellation and callback exceptions still
propagate. These tests validate diagnostics, not recovery success.

## Reproduction

The nine saved 30-second WAVs from the September 14 supplemental run were
replayed without changes to model options, timestamp validation or eligibility.
The completed run is stored under
`benchmarks/runs.local/task10-asr-baseline-20260916/`. Its directory name identifies
the run started before the interruption; completion was inspected September 19.

Evidence SHA-256: `e80450d62c88f56c7d49b6fdf758db1433c64517eac194edcf0fd54136077555`.
Replay command (a repeat must choose a new output directory):

```sh
/usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  /usr/bin/env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/verify_asr_recovery.py \
  --output benchmarks/runs.local/task10-asr-baseline-20260916
```

All five prior failures reproduced at timestamp validation. All nine windows
passed vocal screening and had uniquely top-ranked English language output.
All four previous successes still passed. These are development reproductions,
not a new accuracy benchmark.

| Clip/source | Original source interval | Result | Last returned end in local 30-second WAV |
| --- | --- | --- | ---: |
| F04 — The Almost Perfect Travel Camera.mp4 | 540–570 | failed | 32.00 |
| F04 | 570–600 | failed | 42.80 |
| F04 | 600–630 | passed | 29.98 |
| F05 — Optical Switches Are So Smooth Epomaker SK61 Review.mp4 | 600–630 | failed | 30.04 |
| F05 | 630–660 | failed | 30.28 |
| F05 | 660–690 | passed | 11.90 |
| F06 — 5 Things to Know Before Buying a Cartier Santos.mp4 | 420–450 | passed | 29.94 |
| F06 | 450–480 | passed | 29.98 |
| F06 | 480–510 | failed | 30.90 |

F06's last window also contains an approximately 3.6e-15-second overlap
between adjacent floating-point boundaries (29.900000000000002 versus 29.9).
It independently overruns the end by 0.9 seconds, so rounding alone cannot
recover it. No tolerance/clamping change was made.

F04's second window includes mixed-language/generic trailing text beyond its
real audio duration. That text is suspect, not trustworthy contextual evidence.
The strict guard prevents it entering classification. Merely clipping the end
timestamp would retain potentially invented text and is not a sufficient fix.

Recorded peak process RSS: 1,836,367,872 bytes; MLX allocator peak:
3,069,591,188 bytes. System swap endpoints were both zero. These are overlapping
memory measures and endpoint observations, not values to add or proof of no
transient swap. No new cancellation measurement was made in this replay.

## Upstream observations — not yet a causal fix

The pinned installed MLX-Audio Whisper `generate` implementation can resume
decoding at the last timestamp on a padded trailing audio fragment, and constructs
segment times from generated tokens without enforcing the remaining real audio
duration. This explains how adapter-local times can exceed a correctly sliced
30-second file; it does not prove which decoder change is the best remedy.

Inspection also found that `ApplyTimestampRules.apply` collects token sequence
positions rather than token IDs for its monotonic timestamp mask. A tiny
network-blocked characterization using timestamp_begin=10 and sequence
`[10, 1, 15, 15, 2]` left the earlier timestamp token 12 at logit 0 rather than
negative infinity. This confirms a defect in that installed filter, but no
real-audio A/B test has established that fixing it resolves these five failures.
Neither site-packages nor the pinned dependency lock was modified.
Inspected `decoding.py` SHA-256:
`ab5c5ee8e7389a11e78490abf5ac328ec2609d731ef333c72dbd24ce7ba1724d`.

## Classification labels

This diagnostic pass did not rerun classification and does not claim improved
category accuracy. Prior Task 9 predictions on the retained partial transcripts:

- F04: Cameras (633), secondary Photography (255).
- F05: Computer Peripherals (601), no secondary.
- F06: Men's Jewelry and Watches (581), no secondary.

These remain partial-excerpt results. The original records are preserved.

## Approved filter comparison — completed September 19

The user approved the disposable comparison during this turn. Two sequential,
network-blocked runs used MLX seed 0 and the same nine saved WAVs:

- Baseline: `benchmarks/runs.local/task10-asr-baseline-seed0-20260919/`.
- Filter-only experiment: `benchmarks/runs.local/task10-asr-filter-seed0-20260919/`.
- Disposable script: `(internal, not published) filter_comparison.py`.

Both runs persist Python/package/model-manifest/code identity before inference.
Their replay-script hashes match. The experiment changes only sequence positions
to timestamp token IDs in the process-local filter and records that source delta
in `experimental-filter.json`. It does not edit installed files. The generic
runner's `mode=baseline_no_recovery` denotes absence of recovery; the experiment
sidecar distinguishes its modified filter from the unmodified baseline.

**Result: no improvement.** All nine returned segment records (text and
timestamps) were identical between baseline and experiment. Both runs returned
exit code 1: five failed, four passed. The synthetic monotonic-mask defect was
corrected in the experiment, but this change alone does not solve the observed
audio-boundary failures. No filter change was promoted to production.

Result SHA-256 values:

- Baseline: `3a83c0e27d2fc0d3af0a8bd04c8d4703feae337e2bba6852b34e87dd215aa61c`.
- Experiment: `991cee096b6aedb2cf83540ca5e1b7891e732c1f82541b63942bb6d015bc9cb5`.

Installed `decoding.py` retained the SHA-256 recorded above after the comparison.
The scope of this conclusion is the tested one-line correction, not every
possible upstream timestamp fix. Classification was not rerun because no
transcript changed and failed windows remain ineligible for classification.

## Original experiment proposal and remaining approval gate

The first paragraph below was approved and completed as recorded above. The
shorter-window recovery candidate in the second paragraph remains unapproved.

First isolate the upstream timestamp-filter defect in a disposable comparison,
using the same nine WAVs, pinned weights and strict validators. Change only the
filter logic, retain both raw attempts and compare timing, text and validation
outcomes. Do not patch the installed environment or promote a runtime monkey
patch. If it does not solve boundary overruns, stop that hypothesis rather than
stacking unproven changes.

Only then consider the plan's separate bounded recovery candidate: one retry
pass for timestamp-invalid, English-admitted windows, split into two disjoint
15-second windows. Each subwindow must independently pass screening, language
and timestamps, and all evidence must remain available. This retry policy is
not implemented or approved by the present checkpoint; shorter windows can
still produce overruns and are not assumed to solve them.

## Approved single-pass shorter-window experiment

The user approved one retry pass using two disjoint 15-second subwindows for
each of the five timestamp-invalid English windows. This was a disposable test,
not authorization to integrate a production retry. It used the original,
unmodified MLX-Audio filter, fixed MLX seed 0, the same local pinned models and
network denial. Each of ten subwindows independently passed vocal screening and
English detection; each was then subjected to the unchanged timestamp validator.

Evidence: `benchmarks/runs.local/task10-short-window-once-20260919/results.json`.
SHA-256: `fd247112dd08866465c37d54d43acd3bfe02fb0b30f07be3e956f314bc5b795a`.
Experiment source: `(internal, not published) short_window_probe.py`.
Runtime identity and the fixed selection are persisted before inference.

| Original failed window | First 15 seconds | Second 15 seconds | Whole parent recovered? |
| --- | --- | --- | --- |
| F04 camera, 540–570 | overrun to 16.88 | passed | No |
| F04 camera, 570–600 | passed | passed | Yes |
| F05 keyboard, 600–630 | passed | floating-point overlap | No |
| F05 keyboard, 630–660 | overrun to 15.12 | overrun to 18.96 | No |
| F06 watch, 480–510 | overrun to 15.56 | passed | No |

All end times in the table are local to a 15-second WAV. Five of ten children
passed. Only one of five parents had both children pass. The command correctly
returned exit 1 because not all parents recovered. No further retries occurred.
The recovered F04 parent contains 28.88 seconds of segment support within its
30-second interval; validating timestamps is not proof of perfect transcription.
The retained text concerns camera dust/weather resistance. No new IAB model
classification was run in this ASR-only experiment; earlier category labels
remain the partial-transcript predictions recorded above.

The F05 615–630 child has an end of `12.200000000000001` immediately followed
by a start of `12.2`, an approximately 1.8e-15-second discrepancy. All its
segment boundaries otherwise fit in the 15-second file. This is a distinct
floating-point representation issue, unlike the four genuine child overruns.
It was left rejected in the recorded experiment. Silently calling it passed
would change the agreed validator after seeing the outcome.

Peak process RSS was 1,823,752,192 bytes; MLX allocator peak was 3,069,591,084
bytes. System swap endpoints both measured 1,971,322 bytes (delta zero), not
an assertion about peak swap or process attribution. No new cancellation
latency or category-accuracy claim is made.

**Recommendation:** Do not integrate the 15-second retry as a general solution;
it left four of five parents unresolved. Separate a narrowly specified
floating-point-boundary correction from the substantive audio-duration/decoder
overrun investigation. Any correction must retain rejection of real overlaps
and overruns and must be reviewed/tested before production use. Neither fix
nor a further model experiment was performed by this checkpoint.

## Precision-versus-overrun investigation

An offline examination of saved outputs confirmed that the F05 615–630
overlap is exactly one binary floating-point step (ULP):
`12.200000000000001 - 12.2 = 1.7763568394002505e-15` seconds.
Equivalent timestamp arithmetic reproduces the discrepancy:
`610 * 0.02` yields `12.200000000000001`, while `6.1 + 305 * 0.02`
yields `12.2`. Installed Whisper constructs times using timestamp-token
multiplication plus a seek offset. This is consistent with rounding from
different arithmetic paths, not evidence of overlapping speech.

The original F06 failed window also has a one-ULP adjacent-boundary overlap,
but separately overruns the audio by 0.9 seconds. Every one of the five original
30-second failures still has a genuine overrun. Their seeded baseline overruns
are 2.0, 1.28, 0.04, 0.28 and 0.9 seconds respectively; F04's second window
previously reached 42.8 seconds in the earlier unseeded replay. Do not conflate
the seeded and unseeded raw outputs or imply that every rerun has identical
trailing hallucinated text.

A disposable in-memory simulation snapped an overlapping segment start to the
preceding end only when separated by at most one ULP, then applied the unchanged
production validator. It changed no files or saved transcripts and performed
no model inference. Results:

- Zero of the five original failed 30-second windows recovered.
- In the 15-second retry evidence, one additional child passed; two of five
  parent windows would pass rather than one of five.
- All genuine overruns stayed rejected. Synthetic overlap sizes of 1e-9,
  0.001 and 0.02 seconds remained outside the proposed precision rule at the
  tested boundary.

This demonstrates a narrow precision defect, not a complete ASR solution.
The simulation is not a production implementation or acceptance result.
Recommended implementation scope, requiring approval: normalize only
adjacent local boundary discrepancies at a carefully tested machine-precision
scale, retaining original timestamps in diagnostics; do not loosen the audio
end bound, round everything to centiseconds, or accept meaningful overlaps.
The separate decoder-duration overrun investigation remains necessary.

## Approved precision fix implementation

Commit `f2ec562` implements only the approved adjacent-boundary correction.
For finite positive local starts that overlap the previous end by at most
one ULP, the start is normalized to that previous end, provided the segment
retains positive duration. The original start, corrected start, local end and
segment index are retained in immutable `timestamp_corrections` records on
the persisted window result. Existing records default to no corrections.

The audio endpoint remains exact: even its next representable floating-point
value is rejected when it overruns the actual duration. Genuine overlaps,
negative/non-finite/inverted ranges and corrections that would create zero
duration remain rejected. No models, decoder settings, thresholds or retry
behavior changed.

TDD evidence and implementation report are in
`(internal, not published) precision-report.md`.
Focused tests: 67 passed. Controller full suite: 457 passed; import and dependency
audits passed. Tests include actual JobStore correction-record round trips.

Saved-output replay: `benchmarks/runs.local/task10-precision-revalidation.json`
contains all nine baseline windows and ten shorter-window children, with source
and validator hashes. Only F05 615–630 changed from failed to passed; every
prior passing window stayed passing and all five original 30-second failures
still failed. This was timestamp revalidation, not fresh model inference or
category evaluation. Applied to the experimental retry records, it yields two
passing parents out of five, but that retry is not part of production.

Task 10 remains incomplete because the real duration overruns are unresolved.
Independent scoped review approved the precision fix with no blocking findings.

## Approved word-timestamp alignment experiment

The next approved experiment replayed the same nine seeded 30-second WAVs with
one inference change: MLX-Audio Whisper `word_timestamps` changed from false to
true. The pinned model, seed, English task/language, network denial and strict
production rebase validator were unchanged. The disposable script called the
pinned model directly and did not modify site-packages or production code.

Evidence:

- ASR results:
  `benchmarks/runs.local/task10-word-timestamps-seed0-20260919/results.json`
  (SHA-256
  `763faadbc0b93e8fe31b60dbe393b765a369550230e434149010cd789006f5e6`).
- Experiment source:
  `(internal, not published) word_timestamp_probe.py`.
- Installed `whisper.py` SHA-256:
  `195883b916ab2752b52397f1b57ed4259bf2fdd0c650a4c310d8a5d20dbfa910`.
- Installed `timing.py` SHA-256:
  `71375cdbb8ac0f223a7cdc07ee8238263aaf48825a880cce6f99401af98c6c58`.

All nine windows passed. This recovered all five former timestamp failures and
regressed none of the four passing controls. Every admitted nonblank segment
ended within its 30-second WAV; maximum end times were 29.98, 29.98, 29.90,
29.40, 29.98, 11.84, 29.84, 29.46 and 29.44 seconds in fixed case order.
The precision correction did not need to change any of these outputs.

Total inference time was 26.63 seconds versus 22.24 seconds for the seeded
baseline, a 1.20x ratio on this nine-window sample. Peak process RSS was
1,823,293,440 bytes and MLX allocator peak was 2,476,834,270 bytes. System swap
endpoints were both 1,971,322 bytes; this is not proof of no transient swap.

Only five of nine transcript texts were byte-for-byte identical to baseline.
Two changes look helpful or harmless: camera `senses` became `sensors`, and a
dangling final `And` disappeared from one watch window. Two low-confidence
tails are concerning:

- F04 camera 570–600 ended with `Same as the L10 Healthcare pair is. This is a
  very good idea of the`; its last two raw segments had mean word probabilities
  approximately 0.012 and 0.198.
- F06 watch 480–510 repeated `The Omega Seamaster`; that last segment's mean
  word probability was approximately 0.098 and its final word probability was
  approximately 0.0011.

The current adapter does not retain these diagnostics faithfully in the domain
confidence field: negative Whisper `avg_logprob` is outside the domain's 0–1
range and is replaced with 1.0. Word probabilities are not currently carried
into `TranscriptSegment`. Therefore the successful timestamp result cannot yet
be treated as a safe production recovery policy.

### Same-window semantic comparison

The three complete 90-second word-aligned transcripts were run through the
unchanged MLX embedding retrieval and Qwen2.5 structured classifier. A control
used all baseline decoded text from the same nine WAVs; timestamp-invalid
baseline segments were used only as text for this diagnostic and were not
declared admissible production output. Both runs were offline, serialized and
skipped a new cancellation probe because this comparison targeted semantics.

| Clip | Word-aligned classification | Same-window baseline-text control | Assessment |
| --- | --- | --- | --- |
| F04 camera | primary Cameras and Camcorders (633); secondary Video Production (268) | primary Cameras and Camcorders (633); secondary Photography (255) | Primary stable. Baseline secondary is better supported; word-aligned secondary is a regression. |
| F05 keyboard | primary Computer Peripherals (601); secondary Digital Audio (613) | same | Primary stable. Digital Audio is questionable because this is keyboard acoustics, not audio software; not caused by word alignment. |
| F06 watch | primary Men's Jewelry and Watches (581); no secondary | primary Men's Jewelry and Watches (581); secondary Antiquing and Antiques (240) | Primary stable. Word alignment removes a questionable secondary. |

Both structured runs passed schema validation and released the text model. The
word-aligned results are stored at
`benchmarks/runs.local/task10-word-timestamp-classification-20260919/`
(`classification-results.json` SHA-256
`ca3dfb886374d2d00c4e793662482af701554d6d2fae2d28bf9ca29405239748`).
The baseline-text control is stored at
`benchmarks/runs.local/task10-baseline-text-classification-20260919/`
(`classification-results.json` SHA-256
`779faccb8c74c72bef896fc275bf28815efdddeb860379bfd02276657c7b9d48`).
An interrupted control checkpoint is preserved separately and is not used for
the final comparison.

**Conclusion:** word-level alignment is the first tested option that resolves
all five strict timestamp overruns on this fixed sample, and all three primary
IAB labels remain correct. It is not ready for production promotion because
the sample also demonstrates low-confidence trailing hallucinations and mixed
secondary-label precision. The next bounded investigation should use the saved
word probabilities to define and test a tail-only confidence policy against the
same failures and controls. It must not silently clamp timestamps, loosen the
strict endpoint validator or filter low-confidence text in the middle of an
utterance. Production integration and a fresh cancellation check remain a
separate approval gate.

## Approved saved-output tail-confidence experiment

The user approved the recommended saved-output experiment. It performed no ASR
or audio-model inference. The policy clones the pinned MLX-Audio Whisper
`word_anomaly_score` and `is_segment_anomaly` formulas, verified against the
installed `whisper.py` hash above, and applies them only to a contiguous suffix
of nonblank segments ending within the final two seconds of a 30-second window.
It cannot remove an anomalous segment from the middle of an utterance. It does
not clamp timestamps, loosen the endpoint validator or introduce retries.

Evidence:

- Policy and filtered transcripts:
  `benchmarks/runs.local/task10-tail-confidence-20260920/`.
- `tail-filter-results.json` SHA-256:
  `df128f4504ffecbffbdd37918d8f01aa0f6edb759434a8ba91ab1ae4a2a5f93b`.
- Disposable policy source:
  `(internal, not published) tail_confidence_probe.py`
  (SHA-256
  `0e0b70b91ae8c99f92228eabf368d793655f6b7d550f66ff157e8fd126734693`).

The policy changed two of nine windows and discarded exactly four terminal
segments:

- F04 camera 570–600: three segments from 29.20–29.98 seconds, with mean word
  probabilities 0.0450, 0.0124 and 0.1979.
- F06 watch 480–510: repeated `The Omega Seamaster.` from 29.40–29.44 seconds,
  with mean word probability 0.0983.

Every discarded segment met the pinned anomaly predicate. All other windows
were byte-for-byte unchanged. The genuine `Cardiac Santos.` phrase was retained:
its mean word probability was lower than most speech at 0.439, but its word
timings did not meet that predicate. This is useful sample evidence against a
simple blanket confidence threshold, not proof of general false-positive risk.

### End-to-end classification result

Retrieval and Qwen2.5 classification were rerun offline on the three filtered
90-second transcripts. F04 improved to the expected Cameras and Camcorders
(633) primary plus Photography (255) secondary. F05 was intentionally unchanged
and again returned Computer Peripherals (601) plus questionable Digital Audio
(613), confirming that issue is unrelated to tail filtering.

F06 did not produce an admissible payload. Both bounded attempts selected the
correct Men's Jewelry and Watches primary (581) plus its ancestor Men's
Accessories (580) as a secondary. The unchanged hierarchy validator rejected
both with `HIERARCHY_REDUNDANCY`, so the required classification gate failed.
No category was silently deleted or normalized. The run released the text model;
peak process RSS was 13,800,751,104 bytes and MLX allocator peak was
19,395,460,012 bytes, under the 64 GiB evidence-host ceiling. System swap
endpoints were both zero, not a peak-swap claim.

Classification-results SHA-256:
`147b499945e1284b481aa2b0315a4570fb1731e0bfb59ce11991867b26197a12`.

The unfiltered word-aligned F06 classification had made the same invalid 581 +
580 choice on its first attempt, then corrected to no secondary on attempt two.
To isolate retrieval ordering, a second offline control paired the filtered text
with the original unfiltered candidate order. It still returned 581 + 580 on
both attempts and failed with the same error. This establishes that candidate
reordering is not required to reproduce the failure; the small transcript change
is sufficient to alter correction compliance in this fixed comparison. It does
not imply that the removed repeated phrase was valid evidence.

Factorial-control evidence:
`benchmarks/runs.local/task10-tail-confidence-f06-original-candidates-20260920/`;
`classification-results.json` SHA-256
`4634f934d66957b4642329fe6ffe97bd71a70c3e8a874738e57aa71b9bc8d452`.

**Conclusion:** the saved-output tail policy behaved narrowly and removed only
the identified terminal anomalies on this sample, but the end-to-end gate did
not pass. Do not promote word timestamps or the filter yet. The next blocker is
the classification contract: the JSON schema permits individually valid IDs
whose combination is hierarchically invalid, while post-validation and one
correction retry are not reliable enough to repair every generated pair. A
separate design decision is required before implementation: strengthen
generation so redundant hierarchy combinations are structurally impossible,
or accept classification failure for such windows. The current checkpoint does
neither, and Tasks 11–18 remain paused.

## Approved two-stage classification-contract spike

The user approved a disposable design probe, not a production change. Three
approaches were considered:

1. **Primary first, relational secondary schema (recommended and tested).** A
   small first generation selects only the primary ID. A second full-payload
   generation locks that primary with `const` and constrains
   `secondary_categories` to an enumerated set of zero, one or two categories
   that are pairwise non-overlapping and do not overlap the primary.
2. **One large single-stage relational schema.** This could branch over every
   possible primary and every legal secondary combination, avoiding the extra
   inference but substantially enlarging and complicating the schema. It was
   not needed to answer this spike.
3. **Delete the redundant category after generation.** This is cheapest but was
   rejected because it coerces an invalid model response and violates the
   strict, auditable contract.

The preferred probe reused the tail-filtered F04–F06 transcripts and candidate
shortlists. It performed two temperature-zero generations per case using the
same pinned Qwen2.5 model. The normal strict payload validator remained the
final defense; no hierarchy retry or post-generation category repair occurred.

Evidence:

- Results:
  `benchmarks/runs.local/task10-two-stage-classification-spike-20260920/results.json`
  (SHA-256
  `8fdb641552bbb7af6383481b0caeaacc2c743a9e2cc1c3fc3540b3ad7dcf3765`).
- Disposable source:
  `(internal, not published) two_stage_classification_spike.py`
  (SHA-256
  `22eef1cb80f91e57452979ccadfca1b7caa1cec8311984ce28b122a052cb21ad`).
- Network was denied and the model registry released successfully.

All three cases passed:

| Clip | Stage-one primary | Constrained final classification |
| --- | --- | --- |
| F04 camera | Cameras and Camcorders (633) | Cameras and Camcorders (633) + Photography (255) |
| F05 keyboard | Computer Peripherals (601) | Computer Peripherals (601), no secondary |
| F06 watch | Men's Jewelry and Watches (581) | Men's Jewelry and Watches (581), no secondary |

The F06 ancestor Men's Accessories (580) was absent from every legal secondary
array once primary 581 was locked, so the previously repeated invalid pair was
not generatable. The existing validator accepted all three final payloads.
F05 also stopped emitting the unsupported Digital Audio secondary, although
three cases are not enough to claim a general secondary-precision improvement.

The second-stage schemas were 2,218–2,510 serialized bytes and contained 79–99
legal secondary arrays. Primary generations took 8.60–10.98 seconds and full
payload generations took 20.42–22.71 seconds. Total model generation was 95.56
seconds: 1.08x the failed filtered single-stage run and 1.04x the earlier
successful unfiltered run, both of which spent extra time on F06 hierarchy
retries. Compared with an already-valid one-attempt case, the separate primary
call adds material latency and must be accepted explicitly.

Peak process RSS was 18,910,019,584 bytes; MLX allocator peak was
19,389,037,432 bytes. System swap endpoints were both zero. These remain
endpoint/peak observations within the 64 GiB evidence-host requirement, not a
general capacity proof.

**Recommendation:** adopt the tested two-stage contract rather than response
repair. Stage one should durably select the primary. Stage two should preserve
that exact primary and use canonically ordered, enumerated secondary arrays that
exclude duplicates and every ancestor/descendant pair. The existing validator
must remain unchanged as defense in depth. Cancellation should apply during
both stages and between them; progress should expose the stage boundary.

This spike establishes feasibility only. Production implementation still needs
explicit approval, TDD for schema construction and staged failure/cancellation,
a broader saved-transcript regression set, a fresh real pipeline run, and a
new cancellation measurement. Word timestamps and tail filtering remain
unpromoted until that work passes; Tasks 11–18 remain paused.

## Approved production two-stage classifier

The user approved the recommended design. Production now performs two strict,
temperature-zero constrained generations per analysis window:

1. a primary-only object whose category must belong to the supplied shortlist;
2. a complete payload whose primary is locked with `const` and whose optional
   secondary array is enumerated from only canonical zero-, one- or two-category
   combinations that do not duplicate or overlap the primary or one another.

Both raw stages are durably checkpointed with explicit `primary` and `payload`
stage identifiers before validation continues. Existing saved attempts remain
readable through a backward-compatible default stage. The unchanged strict
payload, shortlist and hierarchy validator remains the final defense. There is
no hierarchy repair, coercion, category deletion or embedding fallback.
Cancellation is checked before generation, after each durable checkpoint and
between stages; a cancellation raised inside either model call is preserved as
an attempt record and propagated.

Production commits are `7222049`, `27a7fbe` and `4ed451f`; intermediate prompt
checkpoint `9d1259a` is preserved as rejected evidence. Deterministic
verification passed **461 tests**, `git diff --check`, the import audit and the
resolved-dependency audit. The focused classifier suite contains 16 tests,
including malformed primary output, locked-primary mismatch, hierarchy defense,
durability before stage two, and cancellation before, during and between stages.

### Preserved prompt investigation

The first eight-case production regression was structurally valid but exposed
a semantic regression: B04 selected Workshops and Classes (247) from an
incidental meetup mention. Its evidence is preserved at
`benchmarks/runs.local/task10-two-stage-saved-regression-20260922/`.
Production result hashes are
`0b09fdc17fa711fa7e71adbdff024a042634e005c918fe828d2027ea684dd4df`
and
`35736dc379277cd9d651bcdb1c40150bbc396569d358396ca4ccc5549e7c2620`.

A first clarification correctly emptied B04 but over-suppressed every optional
secondary in the five-case set, including the substantive Cameras and Video
Gaming contexts. That rejected tradeoff is preserved at
`benchmarks/runs.local/task10-two-stage-saved-regression-v2-20260922/`;
the result hash is
`0ad17447648da9d2243532e518de427f85fbf9025a03a64de9d4eec6a056943f`.

The final instruction distinguishes structural allowance from semantic
relevance and repeats the balanced policy: reject incidental topics, leave
unsupported slots empty, and retain genuinely substantial secondaries. No
third prompt-tuning attempt was made.

### Final saved-input regression

The final production code was run offline against eight previously frozen
English transcript/candidate sets. All eight completed both stages, retained
both raw outputs, passed strict validation without repair and released the text
model:

| Clip | Primary | Secondary |
| --- | --- | --- |
| N01 | Video Production (268) | Cameras and Camcorders (633) |
| N02 | Computer Peripherals (601) | Video Gaming (680) |
| N03 | Men's Jewelry and Watches (581) | None |
| N04 | Men's Jewelry and Watches (581) | None |
| B04 | Computer Peripherals (601) | None |
| F04 | Cameras and Camcorders (633) | Photography (255) |
| F05 | Computer Peripherals (601) | None |
| F06 | Men's Jewelry and Watches (581) | None |

B04 now follows the frozen incidental-meetup policy while N01 and N02 retain
substantive secondary subjects. N04 changed from the prior provisional
Collecting (261) result to no secondary. That label was explicitly deferred for
later review, so this is recorded as an unresolved semantic difference rather
than counted as an accuracy gain or loss. F04–F06 match the approved Task 10
spike outputs.

Evidence:

- `benchmarks/runs.local/task10-two-stage-saved-regression-v3-20260922/task9-production/classification-results.json`
  SHA-256
  `d2d8fbdce9feef3357dcff2fddf73298671b7ecb742b382b26b89a3712f26c06`.
- `benchmarks/runs.local/task10-two-stage-saved-regression-v3-20260922/task9-supplemental/classification-results.json`
  SHA-256
  `0944a8c30100a70a4bdccf12b2eefc3402247e28ae3ed34e4cd19e8247a42f34`.
- Frozen input hashes remained
  `dbbfcd07764673642c983304b69259ed0e4f0f021a38a033b9474066fd5a5b00`
  and
  `74594195dd50cc7b2e7053af51818bf9e48bab0ba9339cdc2aae6b51970d23d9`.

The 16 model calls took **271.80 seconds** in total versus **168.40 seconds**
for the earlier eight one-stage calls, a **1.61x / 61.4%** generation-time
increase on this saved set. This is a material accepted architecture cost, not
an accuracy measurement. Across the two final processes, maximum observed RSS
was 18,539,855,872 bytes and maximum MLX allocation was 19,425,359,664 bytes,
under the 64 GiB evidence-host ceiling. Swap endpoints were zero in both runs.

### Real staged cancellation measurement

A separate one-case production run completed F06 as Men's Jewelry and Watches
(581) with no secondary, then requested cancellation one second into another
two-stage classification. Cancellation interrupted the `primary` generation
and returned in **10.66 seconds**, below the required **35-second** target. The
evidence retained a `primary` attempt with `JobCancelled`, no raw payload and no
fabricated classification. Model release succeeded; peak RSS was
15,967,813,632 bytes, MLX peak allocation was 19,400,834,004 bytes and swap
endpoints were zero.

Evidence:
`benchmarks/runs.local/task10-two-stage-cancellation-20260922/classification-results.json`
(SHA-256
`62d2c16dc91e41d1d9f5825a5bb540eb7294b9351b3fa81d6f5d32bdcf811244`).

**Current conclusion:** the production two-stage classification blocker is
resolved for the tested saved inputs, with auditable live stage progress and a
passing real cancellation target. This does not complete Task 10. Word-timestamp
ASR and the tail-only anomaly filter are still experiments, not production
behavior, and a fresh real media pipeline rerun remains required before their
promotion. Tasks 11–18 remain paused.

## Production promotion and final real-media gate — September 26

This section supersedes the earlier checkpoint conclusions while retaining all
failed and experimental evidence above. The user-approved ASR policy was
implemented in commits `fdafec4` and `1c2c922`, with final review fixes in the
Task 10 follow-up commit `07ded06`:

- request pinned MLX-Audio Whisper word timestamps;
- score the first eight non-punctuation words with the pinned Whisper anomaly
  formula;
- remove only a contiguous anomalous suffix whose segment ends strictly within
  the final two seconds of the actual WAV;
- stop at the first supported or non-terminal segment;
- retain the existing service timestamp validator, English gate, public
  transcriber interface and one-attempt ASR budget.

There is no timestamp clamping, boundary tolerance, JSON/text repair or ASR
retry. Invalid word metadata fails closed. Unit coverage also proves that an
anomalous interior segment and an anomalous segment ending exactly two seconds
before the WAV boundary are retained, and that malformed word metadata fails
closed. The approved word anomaly predicate retains its saved probe semantics;
separately, a segment containing no alphanumeric transcript text is excluded
from semantic evidence, so punctuation-only output cannot make a window
classifiable.

Raw negative, inverted or non-finite segment timestamps now produce the safe
`timestamp_validation/invalid_timestamps` diagnostic at the adapter/service
boundary, including when validation fails before a domain timestamp object can
be constructed. Adjacent overlap correction is restricted to exactly the
single representable float immediately below the previous end; the regression
suite covers a two-step overlap at a binade boundary.

### Final fixed-window replay

The committed adapter replayed the same nine saved WAVs. All five former
timestamp failures recovered and all four controls remained valid:

| Clip | Source interval | Before | Production result | Last retained local end |
| --- | --- | --- | --- | ---: |
| F04 | 540–570 | failed | passed | 29.98 |
| F04 | 570–600 | failed | passed | 29.20 |
| F04 | 600–630 | passed | passed | 29.90 |
| F05 | 600–630 | failed | passed | 29.40 |
| F05 | 630–660 | failed | passed | 29.98 |
| F05 | 660–690 | passed | passed | 11.84 |
| F06 | 420–450 | passed | passed | 29.84 |
| F06 | 450–480 | passed | passed | 29.46 |
| F06 | 480–510 | failed | passed | 29.40 |

The nine language-detection plus ASR calls took 35.30 seconds in this fresh
process. The earlier seeded single-variable comparison remains the valid
overhead comparison: 26.63 seconds with word timestamps versus 22.24 seconds
without, or 1.20x. The final ASR process peaked at 1,850,605,568 bytes RSS and
3,069,591,188 bytes MLX allocation; swap endpoints were zero.

The cancellation timer fired one second into the known longer F04 second
window. The adapter surfaced `JobCancelled` after one generation attempt in
3.27 seconds from request to stop, below the 35-second target. No second ASR
attempt occurred.

Evidence:

- `benchmarks/runs.local/task10-final-asr-20260926/results.json`
  SHA-256
  `cffd00f65a0614e8f38a6379f035d58f73d5fe87676905779a542d0977804cb6`.
- Production adapter SHA-256 recorded before inference:
  `1b30a724ff53442a0eeba0ce6c09b39faaf18ebdfef0e8c940fa30eec2697c65`.
- Replay script SHA-256 recorded before inference:
  `c14c25f19ac8298b3468872597179daa675cd3d8fe26ec0631ec39e388b7947a`.
- Runtime identity records commit `a972eeb`; the worktree was dirty because
  pre-existing unrelated user edits were deliberately preserved. The adapter
  and script hashes bind the executed production paths exactly.

### Fresh three-excerpt audio-only pipeline

The original F04–F06 90-second selections were re-extracted from the video
containers' chosen audio tracks. No video frames were read. Every 30-second
window completed with `attempt_count=1`, no error and monotonic progress
`1/3`, `2/3`, `3/3`. Retained transcript support and final strict labels were:

| Clip | Retained speech support | Transcript span | Primary | Secondary |
| --- | ---: | --- | --- | --- |
| F04 camera | 86.32 s | 0.00–89.90 s | Cameras and Camcorders (633) | Photography (255) |
| F05 keyboard | 64.58 s | 1.68–71.84 s | Computer Peripherals (601) | None |
| F06 watch | 86.26 s | 0.00–89.40 s | Men's Jewelry and Watches (581) | None |

The F05 difference from 90 seconds is silence/non-speech at the excerpt edge,
not an unprocessed or failed ASR window. Classification persisted both raw
stages for every clip, passed strict validation without repair, and took 103.38
seconds across six generations. The required combined gate passed. Its live
primary-stage cancellation probe stopped in 10.22 seconds and retained a
`JobCancelled` attempt without a fabricated payload.

The classification process peaked at 8,082,522,112 bytes RSS and
19,403,586,540 bytes MLX allocation, below the 64 GiB evidence-host ceiling;
swap endpoints were zero. Native Apple Silicon `ffmpeg`/`ffprobe` 9.0.2 was
installed after the first pre-inference initialization attempt correctly
rejected the old Intel-only `/usr/local` binary.

Evidence hashes:

- `fresh-transcription.json`:
  `16e72535d5a78a79bd902b905d64a9b56bbdfb086198cdf1ab986d6f887c3814`.
- `classification-inputs.json`:
  `ad587dbca9a5a3ae47e0f07e6b3d1262c82e884b883d6261ef3ec052e6ee9a32`.
- `classification-results.json`:
  `843d38338890ce49256339ff097959c6db234c6615ed644e70d2c2dc994e3442`.
- evidence protocol:
  `addddb94a393cda46052147bfd59c353b92a95d6eece6c6272e260fbd02f481d`.

The refreshed pipeline directory is
`benchmarks/runs.local/task10-final-pipeline-20260926/`; the ASR replay is
`benchmarks/runs.local/task10-final-asr-20260926/`.

**Final evidence conclusion:** Task 10's production policy reliably recovered
the five reproduced failures on the fixed sample, retained all controls,
completed the original three excerpts end to end with the approved labels, and
met both model-cancellation targets without relaxing validation or adding an
ASR retry. This is bounded evidence for the pinned English/MLX stack, not a
general ASR accuracy claim. Tasks 11–18 remain paused as requested.

Final review: the scoped delta review of `a972eeb..8b68c8b` approved the fixes
with no Critical, Important or Minor findings. Review coverage and exclusions
are recorded in the task conversation; broader semantic accuracy and Tasks
11–18 remain outside this completion claim.

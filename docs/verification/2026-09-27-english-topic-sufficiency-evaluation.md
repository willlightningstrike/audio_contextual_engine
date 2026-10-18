# English topic sufficiency: frozen v1 evaluation

**Status:** **Fail** for the interactive frozen run. Execution and coverage gates failed; qualitative review and resource gates remain pending. This is not a release approval.

## Protocol and provenance

Protocol ID: `english-phase1-topic-sufficiency-v1`. Canonical protocol SHA-256: `ab58fa88518056f3233d84b74d9e98ae986ed53564d4cdf23bf74a7caf26d463`. Manifest and literal validator pin were committed as `63acd9e` before the first v1 acceptance command. The manifest records each original source hash, copied analysis-input hash, exact range, context override, and selection provenance. `validate_protocol` returned no errors; `_verify_inputs` resolved all six files with no hash or duration failures. The full Whitney audio stream was measured at 274.041905 seconds before freezing; the normalized PCM copy measures 274.041938 seconds (33 microseconds of sample rounding).

All six cases are known development material and are explicitly `non_blind: true`; the saved run's blind semantic denominator is zero. Sony and Hario were frozen as topical-positive controls before this attempt. No numeric semantic accuracy is estimated. Earlier frozen manifests and reports are historical and were not rewritten.

## Immutable interactive result

The saved run is `benchmarks/runs.local/english-topic-sufficiency-v1/terminal-run-20260928-01.json` (file SHA-256 `bca0eedfd36400631fc2cb4e2a19a5963f403a0017cf0014429a4fe06c504092`). Its artifacts are under `benchmarks/runs.local/english-topic-sufficiency-v1/terminal-run-20260928-01-artifacts-20260929T004457Z/`. It reports `release_decision: fail`, all six attempted, 68,719,476,736 bytes of host memory, and zero model downloads. The report and artifacts have not been modified or rescored. The earlier `metal-blocked-20260927.json` and `metal-blocked-20260927-v2.json` preflight reports also remain unchanged; they contain no model inference and do not replace this live result.

`T` is usable English transcript duration, `D` is analysed audio duration, and `S(c)` is cited primary-role support. Seconds below are rounded to 0.01 for display; the saved JSON retains full precision. A dash in the whole-file `S(c)` position means no topic-policy decision was recorded, even when a failed job retained window citations. All four completed results have **zero secondary categories**.

| Frozen case | Routing / job | Whole-file outcome | T / D / S(c), seconds | Transcript coverage and error |
| --- | --- | --- | --- | --- |
| `tv-minus01-2-dialogue-600-720` | indeterminate / failed | No final category or insufficiency decision; window-only `163` (Personal Celebrations & Life Events) cited 12.08 s | `21.38 / 120 / —` | Four planned windows attempted; one failed transcription at 600–630 (`NO_USABLE_TRANSCRIPT`); job `ValueError` |
| `tv-minus01-corrective-60-180` | indeterminate / failed | No final decision; window-only `186` (Family and Relationships) cited 7.80 s and `188` (Dating) 20.06 s | `38.08 / 120 / —` | Four planned windows attempted; one failed transcription at 120–150 (timestamp-validation `MODEL_FAILURE`); job `ValueError` |
| `backstreet-vocal-90-120` | accepted English / complete | Primary `363` Rock Music; secondaries none | `16.18 / 30 / S(363)=16.18` | 1/1 window; no technical error |
| `sony-camera-speech-30-90` | accepted English / complete | Primary `514` Hunting and Shooting; secondaries none | `56.90 / 60 / S(514)=56.90` | 2/2 windows; no technical error |
| `hario-coffee-speech-120-300` | accepted English / complete | Primary `222` Non-Alcoholic Beverages; secondaries none | `165.90 / 180 / S(222)=165.90` | 6/6 windows; no technical error |
| `whitney-houston-full-audio` | accepted English / complete job | Primary `186` Family and Relationships; secondaries none; later window primary `339` Adult Contemporary Music | `160.00 / 274.041938 / S(186)=100.48; S(339)=50.28` | 10/10 windows; final 270–274.041938 `VOCAL_PRESENCE_UNCERTAIN`; acceptance execution **failed** and coverage **partial** |

The two TV cases retained window classifications, but their transcription failures prevented a whole-file topic decision. They are execution failures even though the saved processed-window ranges span their frozen excerpts. Whitney's pipeline completed under its terminal uncertain-vocal exception, yet the acceptance report flags its explicit tail error and marks coverage partial: the processed PCM ends at 274.041938 s, 0.000033 s beyond the frozen source end of 274.041905 s. Neither result is converted into a pass here.

## Citation ranges

These are unions of the saved category-specific transcript-segment citations, in absolute source seconds. Adjacent intervals are merged; this display rounds endpoints to 0.01 s. Raw per-segment indices, roles, and full-precision ranges remain in each saved `attempt.category_citations` record.

- Sparse TV, window-only `163`: 665.96–668.42, 673.52–676.80, 677.94–681.08, 703.76–704.82, 716.56–718.70.
- Mixed TV, window-only `186`: 64.82–72.62; window-only `188`: 150.00–156.94, 162.90–166.36, 167.60–170.88, 171.70–175.12, 176.94–179.90.
- Backstreet `363`: 99.46–115.64.
- Sony `514`: 30.00–36.72, 37.08–39.24, 40.72–47.12, 47.72–59.98, 60.00–89.36.
- Hario `222`: 120.00–149.98, 150.00–155.14, 155.30–166.14, 174.44–178.18, 180.00–185.20, 188.50–209.80, 210.00–239.98, 240.00–269.90, 270.00–299.82.
- Whitney `186`: 1.58–18.26, 20.76–29.98, 30.00–36.38, 43.48–54.48, 60.00–62.22, 70.56–73.10, 80.20–83.00, 85.00–88.74, 92.00–104.00, 106.30–116.16, 120.00–123.18, 155.96–170.06, 171.14–175.96, 177.80–179.74. Whitney window-only `339`: 189.54–205.88, 213.28–219.66, 222.18–227.14, 229.02–238.72, 245.76–258.66.

## Attempts and resource measurements

The saved report records 2, 4, 2, 2, 4, and 6 text-generation attempts respectively for the six cases, with zero generation-attempt errors. Values below are per-case observed wall time, process peak RSS, and MLX peak memory; they do not imply semantic correctness. Host swap before, peak, and after was zero for every case.

| Case | Wall time (s) | Peak RSS (bytes) | MLX peak (bytes) | Host memory-pressure peak |
| --- | ---: | ---: | ---: | ---: |
| Sparse TV | 89.07 | 12,547,506,176 | 19,913,587,426 | 52% |
| Mixed TV | 105.89 | 17,268,932,608 | 19,717,534,338 | 53% |
| Backstreet | 60.75 | 17,927,389,184 | 19,519,483,574 | 52% |
| Sony | 83.46 | 14,972,059,648 | 19,897,724,858 | 53% |
| Hario | 148.66 | 18,457,968,640 | 20,184,447,534 | 56% |
| Whitney | 158.44 | 18,345,295,872 | 19,844,902,950 | 54% |

Cold/warm timing was not recorded because the first case failed, and no required live-cancellation stage has evidence in this run. The resource gate therefore remains pending despite the per-case measurements.

## Provisional transcript-grounded assessment

This is an analyst reading of the saved audio-derived transcripts and citations, **not** a `post_result_user_review` approval. The report's `post_result_reviews` array is empty, every attempt says `pending_post_result_review`, and the semantic gate is pending.

- Backstreet: the lyric fragment “rock your body” supplies the cited word but does not establish the acoustic Rock Music genre from transcript alone. The shortlist contained both `349` Contemporary Hits/Pop/Top 40 and `363` Rock Music; neither genre is established by these lyrics. The whole-file `363` label is an unsupported genre inference despite valid JSON and sufficient cited duration.
- Sony: the transcript describes camera screen, composition, menus, bright monitoring, and a camera “shooting page.” It does not discuss hunting or shooting sports. The saved shortlist ranked the correct `633` Cameras and Camcorders first (similarity `0.4353117645`) and `514` Hunting and Shooting fifth (`0.4195270538`); the model selected `514`. This is a model selection/disambiguation error, not missing retrieval. The topical-positive control routed to English and passed the routing gate, but selected the wrong topic.
- Hario: repeated coffee, brewer, filter, water-temperature, and recipe discussion makes the broad `222` beverage category plausible. This is a provisional content assessment, not an approved label.
- Whitney: relationship language makes `186` a plausible lyric-level topic and its recorded `S(186)/T` and `S(186)/D` exceed the frozen 50%/25% cutoffs. The later `339` Adult Contemporary Music window rests on lyric text, not acoustic evidence, and warrants rejection as a transcript-grounded genre claim. The completed job still fails acceptance execution/coverage as recorded.
- Both TV excerpts lack a valid whole-file result because of the named transcription failures. Their cited window labels remain inspectable development evidence, not accepted categories or proof of topic sufficiency.

## Gates and release decision

The saved gates are: protocol integrity **passed**; routing **passed** for the two predeclared topical-positive controls; schema **passed** for completed category outputs; export integrity **passed** for completed exports; execution **failed** for both TV cases and Whitney's explicit tail error; coverage **failed** for Whitney's exact-range mismatch; semantic review **pending**; resources **pending**. The overall saved `release_decision` is **fail**. The schema and routing passes do not override unsupported semantic labels or failed execution/coverage. These six examples are non-blind development evidence, not a measured blind accuracy rate.

No frozen v1 input, cutoff, prompt, manifest, or saved result is revised in response. The independent Task 18 live cancellation, manual UI, and full release gates remain pending. Any future policy or input change requires a new versioned protocol and new evidence.

# Contextual Engine — Capabilities and Limitations Review

**Date:** 2026-10-18
**Scope:** Design documents and source code as they stood before the v3 changes (verifier v1, aggregation v2)
**Method:** Read the design and evaluation documents, inspected the core pipeline, gating, verification and aggregation code, and ran the deterministic test suite.
**Test result:** `pytest -m "not real_models"`: **1,129 passed**, 2 expected warnings, about 29 s.
**Not covered:** Real model inference, model accuracy, live cancellation timing, and manual browser reconnect were not tested in this review. No code was changed.

---

## Summary

Contextual Engine is a local, single-user application for Apple Silicon Macs. It assigns **IAB Content Taxonomy 3.1** topic labels to media, using only the **transcript of English speech or lyrics**. It does not analyse video frames or acoustic genre.

The design is conservative: when the evidence is weak, it returns no label rather than a guess. It is a well-engineered, auditable **experimental prototype**. It works best on long, single-topic English talk content such as reviews, podcasts and tutorials. It is not yet ready for broad production use, for three reasons:

- Only about one file in three receives a label.
- The verifier is the same model as the classifier, so they can share mistakes.
- There is no blind accuracy evaluation.

---

## Implementation status (updated 2026-10-18)

The recommendations in section 4 have since been implemented. Sections 1–3 describe the code as reviewed, before those changes.

| Recommendation | Status |
| --- | --- |
| 4.1 Mixed topics: window winners as secondaries, parent rollup, multi-topic | Done (`english-audio-aggregation-v3`) |
| 4.2 Exclude failed windows with a minimum-coverage rule; disclose partial coverage | Done (20% technical cap; `coverage` export segment) |
| 4.3.1–4.3.2 Schema and token-budget fixes | Done (`english-transcript-verifier-v2`); closes all 31 diagnosed invalid or incomplete generation ranges |
| 4.3.3 Bounded retry | Not added: decoding is greedy, so a retry repeats the same output |
| 4.3.4 Break down the 17 failures | Done: 13 uncited secondaries, 11 verifier truncations, 7 out-of-range indices, 2 ASR timestamp failures |
| 4.4.1–4.4.2 Language and non-vocal gates as exclusions | Done (20% non-English cap; music and silence excluded anywhere) |
| 4.4.3–4.4.5 Independent verifier, blind evaluation, licence | Open |

**New finding.** Replaying v3 over the saved v2 evaluation results changed none of the 99 abstentions. The real bottleneck is that support counts only the segments the verifier explicitly cites. The 93-minute baseball programme had Baseball verified in 61 of 62 windows but only 48% of its speech cited. Crediting whole windows would label about 60 more replayed cases, but also adds some doubtful labels. That choice is left open.

---

## 1. What the application can do

### 1.1 Inputs

| Item | Support |
| --- | --- |
| Formats | `.wav`, `.mp4`, `.mov`, `.m4a`, `.mp3`, `.mkv`, `.webm` |
| Submission | Local path, browser upload (1 GiB default limit), or folder batch |
| Video files | Accepted as containers; only one selected audio stream is analysed |
| Stream choice | An audio stream can be chosen explicitly |
| Context | Defaults to video for video containers and unknown for audio-only files; can be overridden |

### 1.2 Processing pipeline

Source: [`engine/pipeline.py`](../engine/pipeline.py)

```text
Path / upload / folder batch
  → ffprobe: inspect the container and select the audio stream
  → ffmpeg: normalise the selected track
  → YAMNet (MLX): speech/singing eligibility, per 30 s window
  → Whisper large-v3-turbo (MLX): language detection and timestamped transcript
  → Group transcript into ~90 s analysis windows
  → Qwen3 embeddings: shortlist the 15 nearest IAB categories
  → Qwen3.6-35B-A3B: strict JSON classification (primary, secondaries, keywords, entities, sentiment)
  → Qwen3.6-35B-A3B verifier: per-category check against cited transcript segments
  → Completion gate and whole-file topic-sufficiency gate
  → Deterministic aggregation and validated OpenRTB export
```

| Stage | Key behaviour | Code |
| --- | --- | --- |
| Work units | Audio is split into windows of at most 30 s | [`ingest/slice.py`](../engine/ingest/slice.py) |
| Vocal screening | YAMNet speech/singing thresholds decide eligibility | [`models/vocals.py`](../engine/models/vocals.py) |
| Language gate | Only windows whose top detected language is `en` are transcribed | [`transcribe/service.py`](../engine/transcribe/service.py) |
| Analysis windows | Whole segments are grouped into windows of about 90 s | [`transcribe/chunk.py`](../engine/transcribe/chunk.py) |
| Classification | Strict JSON contract: duplicate keys, non-finite numbers and extra fields are rejected | [`signals/classify.py`](../engine/signals/classify.py) |
| Verification | At most 5 calls per window, at most 2 retrieval-ranked alternative primaries, no automatic retry | [`signals/verified_topics.py`](../engine/signals/verified_topics.py), [`signals/topic_verify.py`](../engine/signals/topic_verify.py) |
| Topic policy | The primary must have cited support for **more than 50% of transcript time** and **at least 25% of audio duration** | [`signals/topic_sufficiency.py`](../engine/signals/topic_sufficiency.py) |
| Aggregation | One primary plus up to 2 secondary categories that are not ancestors or descendants of each other; top 20 keywords, top 50 entities, sentiment weighted by duration | [`signals/aggregate.py`](../engine/signals/aggregate.py) |

### 1.3 Output

- One primary IAB 3.0 category and zero to two secondary categories.
- Keywords and named entities. Whole-file sentiment is shown in the browser but not exported.
- An evidence timeline with cited transcript time ranges.
- A coverage summary of analysed, failed and excluded ranges.
- Fields compatible with the OpenRTB content object (`cattax=9`), validated before export.

### 1.4 Service and user experience

- Browser interface at `127.0.0.1:8765` (loopback only).
- Live progress over server-sent events (SSE), showing the active window, its time range and verifier call counts. Progress is saved and never goes backwards.
- Reconnecting resumes from the saved snapshot. After repeated live-update failures, the browser falls back to polling every 5 s.
- Cooperative cancellation at bounded checkpoints, with completed attempts preserved.
- Distinct terminal outcomes: **success**, **insufficient topic evidence**, **technical failure** and **cancelled**. Only successful jobs can be exported.
- Copy JSON and Download JSON return identical server-validated bytes.

### 1.5 Engineering strengths

- **Fails closed.** Malformed model output, citations that point outside the window, and incomplete generation are rejected rather than repaired.
- **Auditable.** Raw generation attempts, prompt hashes, model identity, candidates, verifier attempts and the topic-policy decision are saved for every job.
- **Reproducible and offline.** Models are pinned to exact revisions and checked by hash. A preflight check refuses to start on any mismatch. Model libraries run in offline mode at runtime.
- **Restricted dependencies.** Torch, TensorFlow and Sentence Transformers are prohibited in the runtime environment and enforced by audit scripts.
- **Tested.** 1,129 deterministic tests cover the logic around the models. Seven medium-severity bugs found in an internal review on 2026-10-18 were fixed before this review.

---

## 2. Limitations

### 2.1 Strict gating means most files get no label

In the 308-case development run (non-blind), **96 of 308 cases (31%)** produced a final label. That run used Qwen2.5-32B. A later text-stage replay of 226 of those cases with the current Qwen3.6-35B-A3B labelled 115 against Qwen2.5's 108, with no structured-output failures; the full run has not been repeated.

| Outcome | Cases |
| --- | ---: |
| Final primary label | 96 |
| Insufficient topic evidence | 99 |
| Non-English or unconfirmed language | 61 |
| Uncertain vocal presence | 35 |
| Technical failure | 17 |

The causes are visible in the code:

- **One non-English window ends the job.** Transcription stops at the first window whose detected language is not English ([`transcribe/service.py:277`](../engine/transcribe/service.py:277)), and the completion gate then fails the job. A short foreign-language quote in an otherwise English video is enough.
- **Any window without detected voices blocks completion.** A 30 s instrumental intro, a long pause or a music break causes this ([`completion.py`](../engine/completion.py)). The only exception is a single uncertain **final** window of 30 s or less after windows that were all successful English.
- **Mixed-topic content usually abstains.** No single category can clear the 50% threshold. Secondary categories come **only** from categories cited in a secondary role ([`signals/aggregate.py:198`](../engine/signals/aggregate.py:198)). A category that won the primary role in many windows does not become a secondary category. A programme split between cars and personal finance therefore returns no label rather than "Autos + Personal Finance". A 20-minute editing tutorial in the evaluation abstained for this reason.
- **One failed window blocks the whole file.** A 93-minute baseball programme processed the entire track but could not publish a label because some windows failed verification.

### 2.2 Accuracy and trust

- **The verifier can share the classifier's mistakes.** The classifier and verifier are the same Qwen3.6-35B-A3B model in separate calls. The verifier checks that cited text supports a topic; it does not check whether Whisper transcribed the audio correctly.
- **Known wrong labels.** Short Tissot watch clips were labelled *Remodeling & Construction*, and a camera clip received an incorrect secondary category.
- **Scores are not confidences.** Keyword, entity and sentiment scores are uncalibrated model outputs, and the verifier does not check these fields.
- **No measure of real-world accuracy.** The evaluation was non-blind and used a convenience inventory that included excerpts and near duplicates.
- **Technical failure rate.** 17 of 308 cases failed on invalid structured output, incomplete generation or model errors.

### 2.3 Scope gaps

- **English only.** Non-English and mixed-language media are out of scope.
- **No music genre, mood or style.** IAB Music (338), its descendant genres and Music Video (650) are blocked in code. Instrumental-only audio is out of scope.
- **No video frame analysis.** This is planned for Phase 2.
- **No use of metadata.** Titles, filenames and container type are deliberately not used as topic evidence.

### 2.4 Hardware and performance

| Item | Requirement or measurement |
| --- | --- |
| Platform | macOS 14 or later, Apple Silicon (arm64) |
| Memory | At least 64 GiB unified memory; peak about 22 GiB MLX memory with Qwen3.6 and Qwen3-Embedding-4B, tested on one M1 Max only |
| Disk | About 23 GiB of models; at least 40 GB free for setup |
| Tools | `ffmpeg`, `ffprobe`, `uv` |
| Speed | Median 67.1 s per labelled case; maximum 990 s; longest unsuccessful case 6,526 s (about 1.8 h) |

Smaller machines and guaranteed speed are not established.

### 2.5 Operations and security

- **Single trusted user.** The service has no authentication and is not designed for multiple users or remote access.
- **Sensitive data is kept.** Job records keep transcripts, filenames and diagnostics, and there is no automatic purge of job history.
- **Offline mode is not a firewall.** Offline mode blocks model downloads but does not block the operating system's network.
- **Setup is complex.** YAMNet must be converted in a separate TensorFlow environment before the main setup.

### 2.6 Release readiness

- No source licence has been declared yet.
- Live cancellation limits (35 s for models, 5 s for ffmpeg) are not certified by unit tests.
- Disconnecting and reconnecting a live browser connection has not been verified by hand.
- Tests that use the real models are excluded from the standard suite.

---

## 3. Fit assessment

| Use case | Fit |
| --- | --- |
| Long, single-topic English talk (reviews, podcasts, tutorials) | **Good**, with abstention likely on mixed topics |
| English TV with intros, music breaks or foreign-language segments | **Poor**: often blocked by the language and vocal gates |
| Music, music videos, instrumental content | **Out of scope** |
| Non-English media | **Out of scope** |
| Ad-tech contextual labelling at production scale | **Not ready**: low label rate, slow, no blind accuracy evaluation |
| Auditable research prototype or local analysis tool | **Good** |

---

## 4. Recommended next steps

### 4.1 Mixed-topic content gets no label

1. **Let topics that win windows become secondary categories.** In [`signals/aggregate.py:198`](../engine/signals/aggregate.py:198), secondary candidates come only from secondary citations. Count a category's primary-role support toward its secondary ranking as well. A programme split between cars and money could then produce *Autos* as primary and *Personal Finance* as secondary. This is a small change in one function.
2. **Add an explicit "multi-topic" outcome.** When no category passes 50%, but two or three categories together cover most of the speech and each has at least about 20%, publish them as co-equal topics. Give this outcome a distinct policy version (for example `english-audio-aggregation-v3-multitopic`) so it is never mistaken for a single dominant topic.
3. **Roll up to a shared parent category.** When the window winners share a taxonomy parent, add up their support under that parent. Publish the parent if it clears the threshold. The existing `_ancestors` helper in [`signals/aggregate.py`](../engine/signals/aggregate.py) already supports this. This is likely what the 20-minute editing tutorial needed.
4. **Keep the 50% / 25% rule as the high-confidence bar.** Report it alongside the new outcomes rather than removing it.

### 4.2 One failed window blocks the whole file

1. **Exclude failed windows instead of failing the job.** Treat a verification failure like a language or vocal failure: record it as an excluded range in the coverage summary, and run the whole-file decision on the remaining windows. **Require a minimum share of successful windows**, for example 80% of transcribed time, so the result stays honest. The current blocker is `classification_failed:` in [`completion.py`](../engine/completion.py). This change would let long programmes, such as the 93-minute baseball case, complete.
2. **Publish a "partial coverage" result.** Mark the export with the share of audio covered and list the failed ranges, so downstream users can decide whether to trust it.
3. **Retry a failed window once** before excluding it (see 4.3).

### 4.3 Technical failures (17 of 308)

The text-generation adapter already forces output to match a JSON schema during generation ([`models/text.py`](../engine/models/text.py)). The remaining failures most likely come from two gaps.

1. **Output that runs out of tokens.** *Correction after diagnosis:* the classifier schema already limits keywords and entities to 8 each; the 20 and 50 limits apply only to whole-file aggregation. All 11 truncations were in the verifier, whose fixed 128-token budget could not hold long citation lists. The fix sizes the verifier budget to the window's segment count.
2. **Rules the schema does not enforce.** The verifier schema ([`signals/topic_verify.py:105`](../engine/signals/topic_verify.py:105)) allows any non-negative segment index and allows `supported` with no citations. Both are rejected only after generation. Put the real index limit (one less than the window's segment count) into the schema. Use two schema branches (`oneOf`) so that `supported` requires at least one citation and the other statuses require none. These failures then become impossible outputs.
3. **One bounded retry.** On invalid or incomplete output, retry once with a different seed and a higher token cap. Save both attempts and count retries against the existing limit of 5 verifier calls per window. This relaxes the current no-retry rule, so give it a new policy version.
4. **Break down the 17 failures first.** The raw attempts and `finish_reason` are already stored in the job records, so this is a cheap query. Use it to choose among the fixes above.

### 4.4 Other improvements

1. **Relax the language gate.** Exclude non-English windows from coverage instead of stopping transcription ([`transcribe/service.py:277`](../engine/transcribe/service.py:277)).
2. **Allow non-vocal windows anywhere in the file.** Treat instrumental intros and breaks as excluded ranges, not blockers.
3. **Use an independent verifier**, such as a different model or a check against the audio, so the verifier and classifier do not share mistakes.
4. **Run a frozen, blind evaluation** after these changes. Compare the label rate and the false-label rate against the current 96 of 308.
5. **Declare a source licence** before any public release.

### 4.5 Suggested order

1. **Technical-failure fixes (4.3).** They are cheap and make outputs valid without loosening any rule.
2. **Excluding failed windows (4.2).**
3. **Mixed-topic changes (4.1).**

Steps 2 and 3 change what counts as a valid result. Give each its own policy version and re-evaluate after each.

---

## References

- [README](../README.md)
- [Application overview](application-overview.md)

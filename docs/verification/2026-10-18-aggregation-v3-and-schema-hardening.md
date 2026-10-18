# Aggregation v3 and schema hardening

**Date:** 2026-10-18
**Scope:** The recommendations in section 4 of the [capabilities and limitations review](../capabilities-and-limitations-review.md).
**Status:** Implemented. Not blind-evaluated.

## Summary

| Area | Change | Version marker |
| --- | --- | --- |
| Verifier output | Schema makes out-of-window indices and status/citation mismatches impossible to generate; token budget scales with segment count | `english-transcript-verifier-v2` |
| Classifier citations | Schema forces the primary citation first and lists only valid indices; an uncited secondary is dropped rather than failing the window | (classifier contract; recorded by verifier v2 jobs) |
| Whole-file policy | Failed and non-English windows are excluded within 20% caps; music breaks and silence are excluded anywhere; mixed-topic outcomes added | `english-audio-aggregation-v3` |
| Export | New `contextual_engine` data segments: `aggregation_policy`, `topic_structure`, `coverage` | v3 exports only |

Saved v1/v2 jobs still load, render, and export unchanged.

## 1. Technical failures: diagnosis

All 33 failed ranges in the 17 technical-failure cases of the [v2 evaluation](2026-10-11-transcript-only-v2-evaluation.md) were re-examined from the saved raw attempts.

| Failed ranges | Cause | Fix |
| ---: | --- | --- |
| 13 | Classifier chose a secondary category but cited only the primary | Uncited secondaries are dropped; the verifier still checks every proposal |
| 11 | Verifier listed so many segment indices that it hit its fixed 128-token limit | Budget is now `48 + 4 × segments` (at most 1,024), and the schema caps the list at the window's segment count |
| 7 | Classifier cited index *n* in a window with *n* segments (one past the end) | Valid indices are listed in the schema as an enum, because the decoder ignores integer `maximum` |
| 2 | ASR timestamp failures (`MODEL_FAILURE`) | Not changed; now excluded within the 20% technical cap |

Replaying the 20 saved invalid classifier outputs through the new validator: the 13 uncited-secondary outputs now validate, and the 7 off-by-one outputs can no longer be generated.

**Retry was not added.** The adapter decodes greedily (`make_sampler(0.0)`), so retrying the same prompt reproduces the same output. All diagnosed causes were deterministic schema or budget gaps, now closed.

## 2. Whole-file policy v3

Coverage rules (`engine/completion.py`, `assess_coverage`):

- **Vocal-uncertain windows** (music, silence) are excluded anywhere in the file, with no cap. The v2 rule allowed only one final window of 30 s or less.
- **Non-English windows** are excluded while they are at most 20% of vocal audio. Transcription stops early once non-English audio exceeds 20% of the file, because the cap is then exceeded whatever follows.
- **Technical failures** (ASR and classification) are excluded while they are at most 20% of English speech in scope. The export then reports `coverage = partial_technical_failures`.
- New failure codes: `NON_ENGLISH_SHARE_EXCEEDED`, `TECHNICAL_FAILURE_SHARE_EXCEEDED`, `NO_USABLE_ENGLISH_TRANSCRIPT`, `TRANSCRIPTION_INCOMPLETE`.

Topic rules (`engine/signals/topic_sufficiency.py`, `evaluate_topic_policy_v3`), applied over usable windows only:

1. **Single:** one category's cited primary-role support is more than 50% of usable speech and at least 25% of the audio (the v2 rule, unchanged).
2. **Parent rollup:** otherwise, the most specific taxonomy parent whose descendants together pass the same thresholds.
3. **Multi-topic:** otherwise, two or three non-redundant categories, each at least 20% of usable speech, that together pass the thresholds. They are reported as co-equal.
4. **Secondaries** for single and rollup results now rank on all verified support, so a category that won the primary role in some windows can be a secondary.

## 3. Verification

- Deterministic suite: **1,166 passed** (`pytest -m "not real_models"`), including 29 new tests. Import and dependency audits pass.
- New tests confirm with the real `outlines_core` regex builder that out-of-window indices, empty `supported` citations, and non-primary first citations are rejected by the decoder.
- **Real models, two local clips:**

  | Clip | Result | Earlier saved job |
  | --- | --- | --- |
  | `root-launch-cartier-60s.wav` | Complete, Men's Jewelry and Watches (581), `single`, coverage `complete`, 82 s | Complete, same category |
  | `root-launch-coffee-45s.wav` | `INSUFFICIENT_TOPIC_EVIDENCE`, 81 s | Same |

  All verifier and classifier generations were valid. The two earlier saved jobs (verifier v1, aggregation v2) still load and export.

## 4. Policy replay over the v2 evaluation (estimate, not a re-run)

The v3 rules were replayed over the saved, verified window results of the 308 cases. This doesn't re-run any model, so the schema fixes are not reflected, and cases that stopped at their first non-English window cannot be fully replayed.

| v2 outcome | v3 replay |
| --- | --- |
| 96 classified | 96 single, same primary; 1 gained a secondary (Video Production + Cameras and Camcorders) |
| 99 insufficient | 99 insufficient |
| 17 technical | 1 labeled (watches); 5 insufficient; 11 still over the 20% technical cap (mostly single-window clips, which the schema fixes address instead) |
| 35 vocal uncertain | 1 labeled (Computer Peripherals); 8 insufficient; 25 no usable English transcript; 1 over the technical cap |
| 61 non-English | 60 over the non-English cap; 1 not fully processed |

**Main finding: the mixed-topic rules did not change any of the 99 abstentions.** The bottleneck is not topic fragmentation. Support counts only the segments the verifier explicitly cites, and it cites a subset of each window:

- 50 of the remaining insufficient cases had every window abstain.
- 60 had no topic above 50% of speech, often with nearly every window verified for the same topic. The 93-minute baseball programme had Baseball verified in 61 of 62 windows but only 48% of its speech cited. The 20-minute editing tutorial had all 14 windows classified, but its top topic had only 34% of speech cited.

**Not implemented: window-level credit.** Crediting each window's whole transcript to its verified primary would label about 60 more replayed cases, including the baseball programme (Baseball) and the editing tutorial (Video Production). It would also add doubtful labels, such as "Business Utilities" as a baseball secondary and "Road Trips" from a 5-second clip. This weakens the citation requirement, so it is left as a decision for the maintainer and should be blind-evaluated if adopted.

## 5. Remaining gaps

- No blind evaluation of v3; the replay is an estimate over non-blind data.
- The real-model check covered two short clips, not the 17 technical-failure inputs.
- `scripts/run_pipeline_probe.py` still points at a retired manifest name (`model-manifest.json`) and fails at model initialisation; this pre-dates the change.
- The frozen v2 acceptance protocol (`scripts/transcript_only_acceptance.py`) still names verifier v1 and aggregation v2 by design; a new protocol is needed for a v3 evaluation.

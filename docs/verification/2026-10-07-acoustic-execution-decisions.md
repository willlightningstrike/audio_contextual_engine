# Acoustic feasibility execution decisions

These implementation decisions accompany the [measured feasibility report](2026-10-04-acoustic-model-feasibility.md). They do not admit an acoustic model or alter frozen v1 results. Entries retain the order and substance of the acoustic plan's execution ledger.

1. Treat a MOSS response reaching the token cap as technically incomplete, even if its text parses: the SDK has no explicit finish reason. If too conservative, a complete response at the exact cap is rejected.
2. Keep YAMNet scores diagnostic until a foreground/genre policy is separately reviewed and calibrated. If too conservative, a useful model is withheld.
3. Preserve completed diagnostics in exclusively reserved failure/cancellation reports before propagating the outcome; never report partial success. The cost is extra local failure artifacts and new paths for subsequent attempts.
4. Resolve the development YAMNet assets at `ROOT/data/models/yamnet` and verify their pinned hashes; there is no packaged production manifest at this development path. Relocating the snapshot requires an explicit configuration update.
5. Freeze MOSS greedy temperature zero and a 2,048-token limit before live inference. This allows more reasoning than the default while bounding execution. If inadequate, generations still truncate; increasing the limit would require a separately versioned experiment.
6. Add the selected PCM path alongside its hash to the case manifest, since the path is necessary to locate and verify the normalized input. This adds a required development field; hypothetical older manifests need conversion.
7. Include saved standalone PCM controls with unknown original extraction lineage using only their exact local WAV extent. The cost may be redundant runs; these results cannot establish whole-original-media behavior.
8. Do not use provisional routing gold or unreviewed native-to-IAB mappings to certify foreground/genre admission. Preserve separate nonadmission reasons rather than calling missing review measured inaccuracy. A useful candidate may remain withheld pending reviewed evidence.
9. Preserve byte-distinct historical PCM alongside freshly normalized ranges, deduplicating only exact PCM matches and retaining aliases. The cost is additional runtime/storage for possibly equivalent audio.
10. Reuse the documented TV timestamp normalization only after a filter-only reproduction verifies the duration correction. Timestamp compensation can insert/drop samples and affect music fractions; preprocessing and historical variants remain disclosed.
11. Preserve completed probes and bind them through frozen manifest identity, raw-byte report hashes, and whole-corpus before/after input audits instead of restarting inference for bookkeeping. This is not an atomic per-attempt guarantee and cannot detect malicious change-and-revert; any detected final mismatch is technical and inadmissible.

## Outcome boundary

The frozen corpus has 170 executable inputs and retains 52 unresolved historical references. All 340 model/case pairs have terminal results. MOSS's 67 technical failures, missing reviewed foreground gold/mapping, and YAMNet's diagnostic-only status prevent admission. No production adapter, fusion behavior, or IAB genre mapping was implemented by this feasibility plan.

## Final review scope and dispositions

The final review covers the approved dual-evidence implementation phase beginning at `b08e915`, not the older application. Its explicitly unjudged behaviors remain bounded as follows:

- Whole-file fusion, the 70% promotion policy, runtime persistence/export, and production cancellation remain a later integration plan, as approved.
- Older application code and unrelated dirty files are not certified by this review; they are preserved.
- Acoustic accuracy and native-to-IAB mapping remain unvalidated; neither candidate is admitted.
- The frozen SAM B role contradicts its confirmed no-vocals provenance. The defect is disclosed, not repaired post-result; these results cannot certify negative-control routing. A corrected protocol requires separate review.
- The MOSS prompt/parser asymmetry and token cap remain frozen experiment limitations. Invalid output is not reinterpreted as accepted evidence.
- Missing YAMNet MLX peak and host-pressure measurements, production throughput, and smaller-host compatibility remain unclaimed. RSS is not a substitute.
- Historical inventory completeness relies on the retained inventory audit and execution history; this review checks the declared corpus rather than repeating that historical crawl.
- Input integrity retains only the declared whole-corpus before/after assurance, consistent with decision 11 above.
- Rights/runtime provenance relies on the recorded pinned preflight for this nonadmitted experiment; this is not independent legal advice or new conversion certification.
- Historical test chronology relies on retained execution records; final deterministic tests are rerun separately before completion.
- The preexisting Starlette/httpx deprecation and intentional ASR blank-segment warning remain disclosed rather than changing unrelated dependencies or ASR tests.

The review found no Critical or Important issue. Its two Minor hardening findings (unapproved null metadata and oversized numeric diagnostics) were fixed together in `ed82ed8`, and the single scoped re-review found both addressed with no new breakage. The controller's final `.venv/bin/python -m pytest -q` run on 2026-10-09 passed 960 tests with the two disclosed baseline warnings in 13.70 seconds. Neither finding changes the frozen acoustic outcome or authorizes model reruns.

The current MOSS probe has additional defensive validation after the historical inference revision `826410e`. Those existing reports remain bound to their original script hash; the hardened probe is deterministically tested, not newly live-evaluated. The frozen corpus SHA256 remains `9d01fd0b18550159f67238209339f756d0d41fa3ba2d377167996e57edc9fbe2`, and the older topic-sufficiency v1 report remains `bca0eedfd36400631fc2cb4e2a19a5963f403a0017cf0014429a4fe06c504092`.

Temporary workflow records are preserved locally under `benchmarks/runs.local/acoustic-workflow-audit-20261009/`; models and raw benchmark evidence remain available in their existing paths. This completed plan does not authorize a merge, publication, model deletion, or the deferred runtime/fusion implementation.

# Transcript-only v2 evaluation rerun protocol

This procedure replays the existing **non-blind audio-only** protocol on the same 308 available frozen inputs. It does not refresh the inventory, change labels, inspect video frames, enable genre output, download models, or overwrite the completed run. If inputs, model versions, taxonomy, or policy need to change, create and review a new versioned manifest first.

Frozen manifest: [`acceptance-english-transcript-only-v2.json`](../../benchmarks/acceptance-english-transcript-only-v2.json)

Canonical SHA-256: `c168581ecbe34b7d78f46448407c8c18431abb346b532478016e0e85c9836a2c`

Protocol: `english-audio-transcript-only-v2`; 308 available cases, 51 unavailable inventory inputs; non-blind.

## Requirements

- Use the same Apple Silicon / 64 GiB class of host for comparable resource observations. The recorded run was on an M1 Max with Metal available.
- Preserve the project root separately from the code worktree: media paths in the frozen manifest resolve under the project root, while the runtime code is in `.worktrees/contextual-engine-phase1`.
- Keep the pinned local model snapshots and taxonomy index already under `WORKTREE/data/models`. The run-only bundle script writes manifests pointing to those assets and copies no weights.
- Keep the current ignored run bundle, which contains the standalone server/controller helpers. It is not part of Git. If it has been removed, these exact helper paths will need to be restored from the original run artifact before replay.

## Prepare a fresh run root

Run from Terminal. Choose a new timestamp suffix; do not reuse `transcript-only-v2-evaluation-20261010` or an existing results directory.

```sh
PROJECT="<project-root>"
WORKTREE="$PROJECT/.worktrees/contextual-engine-phase1"
TEMPLATE_RUN="$WORKTREE/benchmarks/runs.local/transcript-only-v2-evaluation-20261010"
RUN_SUFFIX="$(date -u +%Y%m%dT%H%M%SZ)"
EVAL_RUN_ROOT="$WORKTREE/benchmarks/runs.local/transcript-only-v2-rerun-$RUN_SUFFIX"
printf 'Use this exact run suffix in both server/controller terminals: %s\n' "$RUN_SUFFIX"

cd "$WORKTREE"

"$WORKTREE/.venv/bin/python" "$WORKTREE/scripts/transcript_only_acceptance.py" \
  --manifest "$WORKTREE/benchmarks/acceptance-english-transcript-only-v2.json" \
  --source-root "$PROJECT" \
  --output-dir "$EVAL_RUN_ROOT/preflight/freeze-revalidation"

"$WORKTREE/.venv/bin/python" "$TEMPLATE_RUN/preflight/prepare_runtime_bundle.py" \
  --source-models "$WORKTREE/data/models" \
  --target-models "$EVAL_RUN_ROOT/app-data/models"
```

The first command validates media/PCM identities and writes a new freeze record; stop if validation fails. The second command verifies the pinned local ASR, embedding, text, and converted YAMNet manifests and prepares an isolated app-data directory without copying model weights. The server setup gate enables offline mode.

## Start the app and evaluation

Separate Terminal windows do not share shell variables. In **each** window, initialize the same paths first; replace the `RUN_SUFFIX` placeholder with the exact value printed during preparation:

```sh
PROJECT="<project-root>"
WORKTREE="$PROJECT/.worktrees/contextual-engine-phase1"
TEMPLATE_RUN="$WORKTREE/benchmarks/runs.local/transcript-only-v2-evaluation-20261010"
RUN_SUFFIX="YOUR-EXACT-PRINTED-RUN-SUFFIX"
EVAL_RUN_ROOT="$WORKTREE/benchmarks/runs.local/transcript-only-v2-rerun-$RUN_SUFFIX"
cd "$WORKTREE"
```

In Terminal window 1, start the standalone server and leave it running:

```sh
cd "$WORKTREE"
"$WORKTREE/.venv/bin/python" "$TEMPLATE_RUN/preflight/serve_evaluation.py" \
  --source-root "$PROJECT" \
  --data-dir "$EVAL_RUN_ROOT/app-data" \
  --port 18765
```

Wait until the service responds on loopback, then in Terminal window 2 run the exact frozen protocol against that same brand-new results path:

```sh
cd "$WORKTREE"
"$WORKTREE/.venv/bin/python" "$TEMPLATE_RUN/preflight/run_evaluation_standalone.py" \
  --repo-root "$WORKTREE" \
  --source-root "$PROJECT" \
  --run-root "$EVAL_RUN_ROOT" \
  --results-dir "$EVAL_RUN_ROOT/results-standalone" \
  --manifest "$WORKTREE/benchmarks/acceptance-english-transcript-only-v2.json" \
  --freeze "$EVAL_RUN_ROOT/preflight/freeze-revalidation/protocol-freeze.json" \
  --base-url "http://127.0.0.1:18765"
```

The runner checks each selected-audio SHA before submitting the corresponding job and creates its results directory with no-overwrite semantics. It retains progress, transcripts, verifier attempts/citations, final snapshots, resource samples, and successful exports. Let every independent case reach terminal state; a prior case failure must not be used to skip later manifest cases. When all cases finish, stop the server with Ctrl-C.

## Interpretation guardrails

- Keep raw `case-records.jsonl` immutable. The current controller gives `status == failed` precedence over the explicit `INSUFFICIENT_TOPIC_EVIDENCE` code; downstream summaries must normalize abstentions from the explicit error and distinguish them from true range-level technical failures.
- Report one primary and at most two verified secondaries per clip. Otherwise report the exact semantic abstention, out-of-scope/uncertain-vocal disposition, or technical error. Never infer a label from a filename, user correction after inference, visual frame, or an unverified window suggestion.
- Include audio/transcript coverage and the declared `coverage_scope`; 168 inputs have unknown original-media extent and cannot be described as whole-source results.
- Do not calculate population accuracy from this non-blind, mixed-provenance set. The five pre-inference reference clips can be listed as a descriptive spot-check only.
- YAMNet remains limited to speech/singing eligibility. No IAB music/genre output or video-frame inference is part of this Phase 1 protocol.
- Keep the accepted reconnect substitute explicit: API SSE reconnect plus deterministic UI regression is accepted; manual live-browser EventSource transport reconnection remains unverified.

## Run artifacts

The completed run retained its inputs and outputs under `benchmarks/runs.local/transcript-only-v2-evaluation-20261010/`; see the [evaluation report](2026-10-11-transcript-only-v2-evaluation.md). Never edit that frozen input manifest or its completed results in place.

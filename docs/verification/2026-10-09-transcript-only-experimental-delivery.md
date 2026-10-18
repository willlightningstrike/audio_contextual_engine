# Experimental transcript-only delivery check

Checked on 2026-10-10 on the existing `codex/contextual-engine-phase1` branch. Tested implementation commit: `2813a6293c179a0bf04a85f2b9be9f69b523dfb5`, based on Task 2 commit `a8aaf7b60c9fbd25ca0804f8556393881feea3b8`.

Fix-round live checks used base HEAD `336030bc5b316a8ee2a3902d38602d8671783182` plus the user-authorized startup regression/one-line repair included in this fix round.

The deterministic UI/API integration and isolated offline wheel checks passed. Fix-round live checks used a temporary verified bundle assembled entirely from pinned existing local assets, with no downloads or persistent model changes. Full speech, full singing, instrumental exclusion, English-vocal semantic insufficiency, visible partial diagnostics, and actual browser Copy/Download/failed-export checks have now been exercised. **On 2026-10-10 the user explicitly accepted service/API SSE disconnect+reconnect plus the deterministic UI reconnect regression in place of the manual live-browser transport check. Actual browser transport disconnect/reconnect remains unverified.** This narrow acceptance amendment does not establish a browser transport event or waive any other Task 3 requirement. There is no new accuracy or full distribution-readiness claim.

## Delivered behavior

The existing local audio-review layout, system sans/Georgia typography, and paper/surface/ink/teal/coral palette remain in use. No stylesheet or dependencies changed. The source form discloses experimental transcript-only analysis and transcription/error limitations before analysis. The saved-record view displays the server's `evidence_mode`; legacy null-mode records explicitly lack recorded transcript-only verification.

Analysis progress preserves the saved server message (including exact window range, verifier call count, and `starting` state), with the active analysis window identified separately. Saved verification windows display their policy version and distinct supported, abstained, or technical-failure outcomes. Supported primary and optional secondary topics show category names/IDs, window-local zero-based segment indices, and citation time ranges. No client-side category support policy was added; classifier proposals cannot establish verification. There is no acoustic genre result.

Transcripts, model-derived metadata, filenames, and error strings use text nodes. Existing upload, local-path, folder, SSE, cancel, reconnect/polling, timeline, partial-diagnostic, and result flows remain. Cancelled/failed/insufficient records cannot export. Copy JSON and Download JSON both fetch server-validated saved bytes; they do not serialize a browser-side prediction.

## TDD and deterministic gates

Run from `.worktrees/contextual-engine-phase1`:

```sh
.venv/bin/python -m pytest tests/test_static_ui.py -q
```

RED: `6 failed, 9 passed, 1 warning in 1.51s` (exit 1). The failures were missing pre-analysis experimental text, missing transcript-only/legacy mode disclosures, and missing supported/abstained/technical verifier outcomes. The Copy/Download byte regression already passed because that behavior was preserved.

GREEN after the minimal UI change: `15 passed, 1 warning in 1.59s` (exit 0).

Additional behavioral regressions exercise real client execution against an API snapshot, SSE reconnection, stale revision rejection, retained technical diagnostics, cancelled export visibility, and malicious transcript/filename/error text. These are behavioral integration protections for existing flows, not new server policy.

```sh
.venv/bin/python -m pytest tests/test_static_ui.py tests/test_api.py tests/test_sse.py tests/test_offline_integration.py tests/test_setup.py -q
.venv/bin/python -m pytest -q -m 'not real_models'
UV_OFFLINE=1 UV_CACHE_DIR="$PWD/.uv-cache" .venv/bin/python scripts/audit_dependencies.py
UV_OFFLINE=1 .venv/bin/python scripts/audit_imports.py
git diff --check
```

Results: focused `53 passed, 1 warning in 3.29s`; complete checkout model-free suite `1008 passed, 2 warnings in 17.46s`; `Dependency audit passed.`; `Import audit passed.`; final diff check exit 0 with no output. One initial diff check found test trailing whitespace, which was removed before committing.

The warnings are the existing Starlette/httpx deprecation and a `BlankAsrSegmentWarning` in the pre-existing dirty `tests/test_asr_recovery.py`. No failing tests were omitted. No application dependency change or model download was made.

## Isolated offline wheel

Created a fresh directory with `mktemp -d /private/tmp/contextual-task3-archive.XXXXXX`, yielding `/private/tmp/contextual-task3-archive.uA0vrC`. Exported only the committed tree and reused the existing Python environment:

```sh
git archive 2813a62 | tar -x -C /private/tmp/contextual-task3-archive.uA0vrC
ln -s <project-root>/.worktrees/contextual-engine-phase1/.venv /private/tmp/contextual-task3-archive.uA0vrC/.venv
cd /private/tmp/contextual-task3-archive.uA0vrC
UV_OFFLINE=1 UV_CACHE_DIR=<project-root>/.worktrees/contextual-engine-phase1/.uv-cache uv build --wheel
UV_OFFLINE=1 UV_CACHE_DIR=<project-root>/.worktrees/contextual-engine-phase1/.uv-cache .venv/bin/python scripts/smoke_wheel.py
```

Both commands exited 0. Build output: `Successfully built dist/contextual_engine-0.1.0-py3-none-any.whl`. Smoke output: `Installed wheel taxonomy rows: 703; static assets present` and `Wheel installation, imports, static assets, and standards data passed.` The cached build backend was available; no dependencies were downloaded.

ZIP inspection found 62 entries, experimental UI text present, and no `.safetensors`, `.npz`, `.h5`, `.gguf`, `.pt`, or `.pth` weight entries. The checkout's existing `dist/` was not written or replaced.

In that same archive, the focused tests were run with all Python socket connections rejected:

```sh
UV_OFFLINE=1 .venv/bin/python - <<'PY'
import socket, pytest
def deny_network(*args, **kwargs):
    raise AssertionError('offline deterministic checks attempted network access')
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
raise SystemExit(pytest.main(['tests/test_static_ui.py', 'tests/test_api.py', 'tests/test_sse.py', 'tests/test_offline_integration.py', 'tests/test_setup.py', '-q']))
PY
UV_OFFLINE=1 .venv/bin/python -m pytest -q -m 'not real_models'
```

Results: focused `53 passed, 1 warning in 2.97s`; full committed model-free suite `981 passed, 1 warning in 18.06s`. The checkout's additional 27 passing tests come from existing user changes/untracked test files; `git archive` deliberately excludes those. The offline integration uses real ffprobe/ffmpeg, fake model adapters, actual saved API snapshots, local static/taxonomy responses, and byte-identical saved exports. Setup tests cover a missing local model bundle and refusal to construct the server when preflight fails, with network access blocked.

## Host, verified temporary bundle, and live checks

`uname -m` returned `arm64`; a narrowly scoped read-only `sysctl -n hw.memsize` returned `68719476736` bytes (64 GiB). The initial sandboxed hardware query was denied; the read-only escalated query succeeded.

Initial preflight correctly rejected the missing production/YAMNet manifests in the checkout; the default Application Support bundle did not exist. Fix-round investigation found that every existing ASR/embedding/text source passed its pinned inventory loader, and all three loose YAMNet artifacts exactly matched the canonical registry byte counts/SHA-256s. Copied those bytes into `/private/tmp/contextual-task3-live.mgGR1x/prepared/assets`; a temporary manifest uses only the exact existing registry provenance/inventory constants. `load_verified_yamnet_path` accepted it. No conversion was claimed or performed. Standard `fetch_english_phase1(..., download=False)` assembled the temporary production bundle with sockets blocked. A local copy of the existing taxonomy index passed unchanged preflight; a temporary symlink had correctly been rejected. No checkout model/index was modified, and no launch guard was weakened.

The user authorized the startup compatibility repair discovered after preflight: FastAPI 0.141.1 lacks `app.add_event_handler`; the one-line repair registers the same shutdown callback through `app.router.add_event_handler`. Regression RED failed with that exact AttributeError (`1 failed, 1 warning in 0.76s`). GREEN setup/offline tests: `16 passed, 1 warning in 2.11s`. Full model-free suite: `1009 passed, 2 warnings in 19.82s`. The actual production service calls normal `scripts.serve.main()` under the temporary data root, offline environment, and non-loopback socket rejection. After the controller's resource-contention instruction, further broad CPU reruns were deferred until all live cases terminalized. Recorded live wall times are observed acceptance timings, not controlled performance benchmarks.

Final pre-commit reruns after the acceptance amendment: `.venv/bin/python -m pytest tests/test_setup.py tests/test_offline_integration.py tests/test_static_ui.py tests/test_sse.py tests/test_api.py -q` → `54 passed, 1 warning in 3.48s`; `.venv/bin/python -m pytest -q -m 'not real_models'` → `1009 passed, 2 warnings in 15.92s`; `git diff --check` and `git diff --cached --check` exited 0 with no output. Existing warning categories remain unchanged.

Native Chrome computer control used the visible local-path form, audio-track probe, and Start analysis. Source/runtime/output data below are actual observations, not inferred accuracy. Complete temporary JSON snapshots, citation ranges, and revision histories are saved under `/private/tmp/contextual-task3-live.mgGR1x`.

| Requested live check | Status for this delivery check |
| --- | --- |
| Full English speech clip | Original Sony A7Cii MP4, 364.367528 s; job `6ec77846-91c3-4a14-8971-834284e939bd`, 378.487995 s wall. Three supported camera windows (last also Photography), one technical INVALID_CLASSIFICATION_OUTPUT window, final 4.37 s language-detection failure; terminal failed ValueError with retained partial diagnostics. |
| Full English singing clip | Complete existing Whitney WAV, 274.041938 s; job `be238773-76d8-41f4-9f8f-c7079890daeb`, 237.724333 s wall. Complete; two supported Family and Relationships windows and one abstained window, uncertain terminal vocal range excluded; no acoustic genre. |
| Instrumental-only out-of-scope clip | Existing `vocal-review-v3/T04.wav`, exactly 5 s; job `685bcbad-efd5-4537-b1db-5c6e90b83366`. Excluded VOCAL_PRESENCE_UNCERTAIN, zero transcript/classifications, terminal failed ValueError, export409. Uncertainty is not claimed to prove no vocals. |
| Unsupported-topic English vocal clip | Existing `backstreet_90-120.wav`, complete 30 s; job `7e3626c0-d015-463d-9def-a4733d9d8441`, 74.302356 s wall. Actual English singing transcription 16.18 s, no failed/excluded ranges, verifier abstained INSUFFICIENT_TOPIC_EVIDENCE/no citations; export 409. Lyric words do not establish Rock Music. |
| Cancellation during verification, request-to-stop ≤35 s | Real service-owned Sony60 job `ff38e66f-eea8-4c50-acbf-f4cc11845d86`: POST cancellation after verifier started, request-to-saved-cancelled 14.738865082996199 s, within 35 s; browser CANCELLED, retained transcript, export 409. This is an observed upper bound on generation stop, not worker-thread shutdown timing. |
| Browser disconnect/reconnect with live inference | UNVERIFIED. Offline emulation did not sever the existing SSE. Normal reload cannot recover this client's non-persisted job identifier. User explicitly accepted the real API/SSE reconnect plus deterministic UI reconnect regression as the Step 4 substitute on 2026-10-10; no browser transport event is claimed. |
| Real API/SSE disconnect/reconnect | Actual socket close/new connection during singing recovered revisions 82/82/82, mode/policy/supported citations, monotonic true. This is not browser EventSource acceptance. |
| Visible live partial diagnostics | Browser displayed Sony supported primary/secondary citation indices/time ranges, separate technical-failure card, transcript/coverage/language failure, and export unavailable. Singing/negative cases showed separate semantic insufficiency. |
| Failed-result export and Copy/Export bytes in browser | Failed Sony export visibly returned completion-only rejection; API 409. Actual Whitney Copy clipboard, Download artifact, API export and persisted result all equal 355 UTF-8 bytes; SHA-256 `aba3eb24e2573303bfe041bab513904cd482ff0aa795c70bc5883aa78ab4f428`. |

For Copy, pasted the actual system clipboard into the local path field without submitting and compared the entire 355-byte ASCII value. Download finished in Chrome's existing default Downloads directory, then only that new test artifact was moved with Finder into the temporary evidence root and compared byte-for-byte. No browser preferences, existing downloads, media, or OS permissions were changed.

An earlier direct Pipeline cancellation probe was invalidated by accidental concurrent store recovery (`interrupted_by_restart`). Its 13.408870207989821 s is explicitly excluded, not a passing cancellation claim. The service is now the sole store owner. An attempted DevTools console helper failed before any EventSource close/dispatch; no mutation executed, and all such attempts stopped. Actual browser reconnect remains unverified; the user's explicit narrow substitute resolves that Task 3 acceptance item without changing what was observed.

## Measured real ffmpeg cancellation

Exercised production `engine.ingest.audio.prepare_audio` with the existing local `benchmarks/runs.local/english-topic-sufficiency-v1-inputs/whitney_full.wav`. A worker thread used the real ffprobe and real ffmpeg process. A thin spawn observer set an event immediately after the real `subprocess.Popen`; the command and production wait/termination code were unchanged. Once the spawned process was confirmed running, recorded `time.monotonic()`, called `CancellationToken.cancel()`, and joined the extraction worker with a 5-second limit. All normalized/partial files were confined to a new `task3-ffmpeg-*` temporary directory.

Observed output:

```json
{"ffmpeg_running_at_request": true, "request_to_worker_stop_seconds": 0.054656249994877726, "worker_stopped": true, "process_returncode": 255, "outcome": ["cancelled"], "partial_audio_exists": false}
```

This measured extraction cancellation meets the 5-second ffmpeg limit for that operation. It does not measure model cancellation or certify all media/host conditions.

## Acceptance amendment and remaining limitations

The user-approved substitution resolves the previously open reconnect acceptance item for Task 3: real service/API SSE socket close/reconnect with saved-snapshot recovery plus the passing deterministic UI reconnect regression. No actual browser transport disconnect/reconnect was verified. Task 3 is ready for independent review under that amended scope; Task 4 has not been started. The release remains experimental. Sony's full-media run retained technical/language failures, semantic labels are not independently certified, cancellation timings cover only the measured operations, and no full distribution-readiness claim is made.

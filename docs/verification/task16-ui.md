# Task 16 UI Verification

Date: 2026-09-27

## Automated checks

- `.venv/bin/python -m pytest tests/test_static_ui.py tests/test_api.py tests/test_sse.py -q`
  — 26 passed.
- `.venv/bin/python -m pytest -q` — 581 passed.
- `node --check engine/web/static/app.js`, `git diff --check`, and
  `git diff --cached --check` — passed.
- The page is served at `/` with local-only `/static` assets; taxonomy names,
  audio-window outcomes, and server-computed coverage are exposed through the
  API projection and SSE snapshot.

## Browser exercise

Used a disposable local FastAPI service, temporary synthetic `.wav` files, two
fake audio streams, and a model-free runner. No production models were loaded,
and no project media was sent to an external service.

- Local-path probing displayed both tracks. Selected track 1 and context
  `Unknown`; the job request persisted `audio_stream_index: 1` and context `7`.
- Progress advanced from queued through extraction, English eligibility,
  transcription, classification, and completion. The completed page displayed
  the whole-file result separately from the timeline, plus keywords, entity,
  sentiment, and transcript detail.
- A transcript containing `<img src=x onerror="window.__xss=true">` appeared as
  literal text; the rendered transcript contained zero `img` elements.
- Copy reported success; Download reported success. Both actions call the same
  `exportBytes()` helper, which reads the same validated `/export` response.
  The browser automation did not expose a downloadable object-URL file path or
  clipboard readback, so a direct byte-for-byte capture comparison was not
  available in this exercise.
- Cancellation was requested while the synthetic run was transcribing. The UI
  announced the request, then showed the terminal cancelled state and stated
  that partial evidence remains while export is unavailable.
- A two-clip batch completed with one successful clip and one failed clip; the
  page displayed both filenames, statuses, and the successful clip's categories.
- The successful upload chooser-to-result flow was not completed in the browser;
  upload creation is covered by API/contract tests, while browser verification
  exercised the local-path workflow.

Independent review follow-up: successful result diagnostics are now projected
into the timeline while a job is partial; progress exposes the lifecycle stage;
coverage says how many windows are recorded rather than implying an unknown
final denominator; the large results tree is not an aria-live region; and the UI
starts monitoring immediately while taxonomy names load in parallel, showing a
visible ID-only warning if taxonomy loading fails.

## Limits

This was a UI/API integration exercise with synthetic, model-free job data, not
an actual media/model accuracy test. A deliberate SSE disconnection/reconnect
and out-of-order revision injection were not run in the browser. The client
implements the five-second polling fallback after repeated SSE errors and
rejects lower revisions; these are covered by focused source/API contracts, but
remain candidates for a later end-to-end browser test.

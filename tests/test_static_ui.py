from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from engine.store import JobStore
from engine.web.app import create_app

STATIC = Path(__file__).parents[1] / "engine" / "web" / "static"


class PageContract(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.inputs: dict[str, dict[str, str | None]] = {}
        self.labels: set[str] = set()
        self.label_targets: set[str] = set()
        self.buttons: set[str] = set()
        self.regions: dict[str, str | None] = {}
        self.text: list[str] = []
        self._label_text = False
        self._button_text = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag in {"input", "select", "textarea"} and values.get("id"):
            self.inputs[values["id"]] = values
        if tag == "label":
            self._label_text = True
            if values.get("for"):
                self.label_targets.add(values["for"])
        if tag == "button":
            self._button_text = True
        if values.get("id") and values.get("aria-live") is not None:
            self.regions[values["id"]] = values.get("aria-live")

    def handle_endtag(self, tag: str) -> None:
        if tag == "label":
            self._label_text = False
        if tag == "button":
            self._button_text = False

    def handle_data(self, data: str) -> None:
        self.text.append(data)
        if self._label_text and data.strip():
            self.labels.add(data.strip())
        if self._button_text and data.strip():
            self.buttons.add(data.strip())


def _page() -> tuple[PageContract, str]:
    source = (STATIC / "index.html").read_text(encoding="utf-8")
    parser = PageContract()
    parser.feed(source)
    return parser, " ".join(parser.text).casefold()


def test_page_provides_keyboard_labeled_upload_path_and_batch_choices() -> None:
    parser, _ = _page()
    expected = {"source-mode-upload", "source-mode-path", "source-mode-batch", "source-file", "source-path", "source-directory", "audio-stream", "content-context"}
    assert expected.issubset(parser.inputs)
    assert {"Upload a file", "Use a local file path", "Analyze a folder"}.issubset(parser.labels)
    assert expected.issubset(parser.label_targets)


def test_page_explains_english_scope_without_free_language_override() -> None:
    parser, text = _page()
    assert "english speech" in text
    assert "usable english lyrics" in text
    assert "instrumental" in text and "non-english" in text
    assert not any("language" in (attrs.get("name") or "").casefold() for attrs in parser.inputs.values())


def test_page_separates_whole_file_prediction_from_timeline_and_diagnostics() -> None:
    _, text = _page()
    for phrase in (
        "primary category",
        "secondary categories",
        "keywords",
        "named entities",
        "sentiment",
        "timeline",
        "coverage",
        "failed ranges",
        "per-clip outcomes",
    ):
        assert phrase in text
    assert "genre" not in text and "mood" not in text


def test_page_has_live_status_and_explicit_cancel_copy_and_export_controls() -> None:
    parser, _ = _page()
    assert {"job-status", "progress-status", "current-stage"}.issubset(parser.regions)
    assert "polite" in parser.regions.values()
    assert {"Cancel", "Copy JSON", "Download JSON"}.issubset(parser.buttons)
    assert "job-results" not in parser.regions


def test_page_loads_local_assets_without_remote_dependencies() -> None:
    source = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "/static/styles.css" in source
    assert "/static/app.js" in source
    assert "https://" not in source
    assert (STATIC / "styles.css").is_file()
    assert (STATIC / "app.js").is_file()


def test_client_renders_untrusted_result_strings_as_text_and_shares_export_bytes() -> None:
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in source
    assert "textContent" in source
    assert source.count("await exportBytes()") == 2
    assert "navigator.clipboard.writeText" in source
    assert "new Blob([bytes]" in source
    assert "revision < state.latestRevision" in source
    assert "await taxonomyReady" not in source
    assert "category names are unavailable" in source.casefold()
    assert "loading category names from the local taxonomy" in source.casefold()
    assert "windows recorded" in source


_DOM_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.className = '';
    this.hidden = false;
    this.style = {};
    this.dataset = {};
    this._text = '';
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this._text = ''; this.children = items; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener() {}
  get options() { return this.children; }
}
const elements = new Map();
const document = {
  createElement: tag => new Element(tag),
  getElementById: id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  },
  querySelectorAll: () => [],
};
const context = vm.createContext({
  document, console, fetch: async () => ({ok: true, json: async () => []}),
  setInterval: () => 1, clearInterval: () => {},
});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const call = (name, ...args) => vm.runInContext(name, context)(...args);
const snapshot = JSON.parse(process.argv[2]);
vm.runInContext('state.itemKind = "job"; state.itemId = "saved-job"', context);
call('renderSnapshot', snapshot, true);
const get = id => document.getElementById(id);
const articles = () => get('timeline-content').children.map(item => ({
  text: item.textContent,
  headings: item.children.flatMap(child => child.children.filter(grandchild => grandchild.tag === 'h3').map(grandchild => grandchild.textContent)),
}));
const first = {
  status: get('job-status').textContent,
  timeline: articles(),
  transcript: get('transcript-content').textContent,
  transcriptTags: get('transcript-content').children.flatMap(item => item.children.map(child => child.tag)),
  abstained: get('abstained-ranges').textContent,
  failed: get('failed-ranges').textContent,
  excluded: get('excluded-ranges').textContent,
  copyHidden: get('copy-button').hidden,
  downloadHidden: get('download-button').hidden,
};
get('job-status').textContent = 'Connection interrupted';
call('renderSnapshot', snapshot, true);
const reconnect = {
  status: get('job-status').textContent,
  timeline: articles(),
  transcript: get('transcript-content').textContent,
  copyHidden: get('copy-button').hidden,
  downloadHidden: get('download-button').hidden,
};
call('renderSnapshot', {
  ...snapshot, status: 'complete', error: null,
  progress: {...snapshot.progress, revision: 8, stage: 'complete'},
  ortb_content: {cat: ['633'], context: 6, kwarray: [], data: []},
  topic_abstentions: [],
  coverage: {...snapshot.coverage, abstained_ranges: []},
});
const complete = {copyHidden: get('copy-button').hidden, downloadHidden: get('download-button').hidden};
call('renderSnapshot', {
  ...snapshot, status: 'cancelled', error: null,
  progress: {...snapshot.progress, revision: 9, stage: 'cancelled'},
});
const cancelled = {copyHidden: get('copy-button').hidden, downloadHidden: get('download-button').hidden};
call('renderBatchChildren', [{
  id: 'child', status: 'failed', error: 'INSUFFICIENT_TOPIC_EVIDENCE',
  request: {source: '/media/child.wav'}, ortb_content: null,
}]);
const batch = get('batch-outcomes').textContent;
process.stdout.write(JSON.stringify({first, reconnect, complete, cancelled, batch}));
"""


def test_insufficient_topic_snapshot_renders_partial_evidence_and_reconnects_without_export() -> None:
    if not shutil.which("node"):
        pytest.skip("Node.js is required for the browser-script DOM contract")
    snapshot = {
        "id": "saved-job", "status": "failed", "error": "INSUFFICIENT_TOPIC_EVIDENCE",
        "progress": {"revision": 7, "stage": "failed", "percent": 100, "message": "Saved"},
        "ortb_content": None, "topic_policy": {"outcome": "insufficient"},
        "text_timeline": [{
            "time_range": {"start_seconds": 10, "end_seconds": 12},
            "payload": {"primary_category": "633", "secondary_categories": []},
            "evidence_ranges": [{"start_seconds": 10, "end_seconds": 12}],
        }],
        "topic_abstentions": [{
            "time_range": {"start_seconds": 0, "end_seconds": 4},
            "error_code": "INSUFFICIENT_TOPIC_EVIDENCE",
        }],
        "transcript": [{
            "time_range": {"start_seconds": 0, "end_seconds": 4},
            "text": "<img src=x onerror=alert(1)> spoken words",
        }],
        "coverage": {
            "total_windows": 2, "successful_windows": 1, "transcribed_seconds": 4.0,
            "classified_units": 1, "failed_ranges": [], "excluded_ranges": [],
            "abstained_ranges": [{
                "time_range": {"start_seconds": 0, "end_seconds": 4},
                "error_code": "INSUFFICIENT_TOPIC_EVIDENCE",
            }],
        },
    }
    result = subprocess.run(
        ["node", "-e", _DOM_HARNESS, str(STATIC / "app.js"), json.dumps(snapshot)],
        capture_output=True, text=True, check=True,
    )
    rendered = json.loads(result.stdout)
    for view in (rendered["first"], rendered["reconnect"]):
        assert view["status"].startswith("Not enough topical evidence in the audio.")
        assert "set Content context to Music" in view["status"]
        assert len(view["timeline"]) == 2
        assert view["timeline"][0]["text"] == "0:00–0:04No supported IAB topic"
        assert view["timeline"][0]["headings"] == []
        assert "Category 633" in view["timeline"][1]["text"]
        assert "1 transcript segment supported this window." in view["timeline"][1]["text"]
        assert "<img src=x onerror=alert(1)> spoken words" in view["transcript"]
        assert view["copyHidden"] and view["downloadHidden"]
    assert rendered["first"]["transcriptTags"] == ["time", "span"]
    assert rendered["first"]["abstained"] == "0:00–0:04No supported IAB topic"
    assert rendered["first"]["failed"] == ""
    assert rendered["first"]["excluded"] == ""
    assert rendered["complete"] == {"copyHidden": False, "downloadHidden": False}
    assert rendered["cancelled"] == {"copyHidden": True, "downloadHidden": True}
    assert "child.wav" in rendered["batch"]
    assert "No supported IAB topic" in rendered["batch"]


def test_fastapi_serves_page_and_assets_from_local_static_bundle(tmp_path: Path) -> None:
    class Runner:
        async def enqueue(self, item_id: str) -> None:
            del item_id

        async def cancel(self, item_id: str) -> None:
            del item_id

    class Settings:
        data_dir = tmp_path / "data"
        allowed_roots = (tmp_path,)
        max_upload_bytes = 20
        allowed_suffixes = frozenset({".wav"})

    client = TestClient(
        create_app(store=JobStore(Settings.data_dir / "jobs"), runner=Runner(), settings=Settings)
    )
    page = client.get("/")
    assert page.status_code == 200
    assert "Contextual Engine" in page.text
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def _run_ui(script: str, snapshot: dict | None = None) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node.js is required for the browser-script DOM contract")
    # Use the real client; only the browser DOM/network boundary is substituted.
    harness = _DOM_HARNESS.split("const snapshot = JSON.parse(process.argv[2]);")[0]
    result = subprocess.run(
        ["node", "-e", harness + script, str(STATIC / "app.js"), json.dumps(snapshot)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


_ASYNC_UI_SETUP = r"""
const get = id => document.getElementById(id);
const readState = () => vm.runInContext('state', context);
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => {resolve = yes; reject = no;});
  return {promise, resolve, reject};
};
const response = value => ({ok: true, json: async () => value});
const job = (id, revision = 1, status = 'analysing', category = null) => ({
  id, status, error: null, progress: {revision, stage: status, percent: 50, message: id},
  ortb_content: category ? {cat: [category], context: 6, kwarray: [], data: []} : null,
  coverage: null, transcript: [], text_timeline: [], topic_abstentions: [], verification_windows: [],
});
const batch = (id, revision = 1, status = 'running', children = []) => ({
  id, status, progress: {revision, stage: 'batch_processing', percent: 50, message: id},
  child_job_ids: children, child_outcomes: {}, coverage: null,
});
const probe = {audio_streams: [{index: 0, codec_name: 'pcm_s16le', language: 'eng'}], selected_audio_stream: 0};
const sources = [], timers = new Map(), requests = [];
let timerId = 0;
context.EventSource = class {
  constructor(url) {this.url = url; this.listeners = {}; this.closed = false; sources.push(this);}
  close() {this.closed = true;}
  addEventListener(name, listener) {this.listeners[name] = listener;}
};
context.setInterval = callback => {timers.set(++timerId, callback); return timerId;};
context.clearInterval = id => timers.delete(id);
context.fetch = async url => {requests.push(url); throw new Error('Unexpected request: ' + url);};
// Let the startup taxonomy request settle before manipulating run state.
await new Promise(resolve => setImmediate(resolve));
"""


def _run_async_ui(script: str, parameters: dict | None = None) -> dict:
    return _run_ui(
        "(async () => {\n" + _ASYNC_UI_SETUP + script
        + "\n})().catch(error => {console.error(error); process.exitCode = 1;});",
        parameters,
    )


@pytest.mark.parametrize("kind", ["job", "batch"])
@pytest.mark.parametrize("outcome", ["success", "error"])
def test_delayed_run_refresh_cannot_replace_or_stop_a_new_run(kind, outcome) -> None:
    rendered = _run_async_ui(r"""
const {kind, outcome} = JSON.parse(process.argv[2]);
const make = kind === 'job' ? job : batch;
const pending = deferred();
context.fetch = url => {requests.push(url); return pending.promise;};
call('beginRun', make('old-run'), kind);
const refresh = call(kind === 'job' ? 'refreshJob' : 'refreshBatch');
call('beginRun', make('new-run', 2), kind);
const currentSource = readState().eventSource, currentTimer = readState().pollTimer;
if (outcome === 'error') pending.resolve({ok: false, json: async () => ({detail: 'Old request failed'})});
else pending.resolve(response(make('old-run', 40, 'complete', kind === 'job' ? '633' : [])));
await refresh;
process.stdout.write(JSON.stringify({id: readState().latestSnapshot.id, revision: readState().latestRevision,
  progress: get('progress-status').textContent, notice: get('job-status').textContent,
  copyHidden: get('copy-button').hidden, downloadHidden: get('download-button').hidden,
  active: kind === 'job' ? !currentSource.closed && readState().eventSource === currentSource
    : timers.has(currentTimer) && readState().pollTimer === currentTimer}));
""", {"kind": kind, "outcome": outcome})
    assert rendered["id"] == "new-run"
    assert rendered["revision"] == 2
    assert rendered["progress"] == "new-run"
    assert rendered["notice"] == ""
    assert rendered["copyHidden"] and rendered["downloadHidden"]
    assert rendered["active"]


def test_delayed_batch_children_cannot_replace_new_children_or_stop_new_polling() -> None:
    rendered = _run_async_ui(r"""
const pending = deferred(), requested = deferred();
context.fetch = async url => {
  if (url === '/api/batches/old-batch') return response(batch('old-batch', 40, 'complete', ['old-child']));
  if (url === '/api/jobs/old-child') {requested.resolve(); return pending.promise;}
  throw new Error('Unexpected request: ' + url);
};
call('beginRun', batch('old-batch'), 'batch');
const refresh = call('refreshBatch');
await requested.promise;
call('beginRun', batch('new-batch', 2), 'batch');
call('renderBatchChildren', [{...job('new-child'), request: {source: '/media/new.wav'}}]);
const currentTimer = readState().pollTimer;
pending.resolve(response({...job('old-child', 40, 'complete', '633'), request: {source: '/media/old.wav'}}));
await refresh;
process.stdout.write(JSON.stringify({children: readState().latestBatchChildren.map(child => child.id),
  text: get('batch-outcomes').textContent, active: timers.has(currentTimer), timer: readState().pollTimer}));
""")
    assert rendered["children"] == ["new-child"]
    assert "new.wav" in rendered["text"] and "old.wav" not in rendered["text"]
    assert rendered["active"] and rendered["timer"] is not None


def test_overlapping_batch_refresh_keeps_children_from_the_newest_snapshot() -> None:
    rendered = _run_async_ui(r"""
const pending = deferred(), requested = deferred();
let refreshCount = 0;
context.fetch = async url => {
  if (url === '/api/batches/same-batch') return response(batch('same-batch', ++refreshCount + 1, 'running', ['child']));
  if (refreshCount === 1) {requested.resolve(); return pending.promise;}
  return response({...job('child', 3, 'complete', '255'), request: {source: '/media/new.wav'}});
};
call('beginRun', batch('same-batch'), 'batch');
const oldRefresh = call('refreshBatch');
await requested.promise;
await call('refreshBatch');
pending.resolve(response({...job('child', 2), request: {source: '/media/old.wav'}}));
await oldRefresh;
process.stdout.write(JSON.stringify({revision: readState().latestRevision, text: get('batch-outcomes').textContent}));
""")
    assert rendered["revision"] == 3
    assert "new.wav" in rendered["text"] and "Category 255" in rendered["text"]
    assert "old.wav" not in rendered["text"]


@pytest.mark.parametrize("callback", ["message", "malformed", "error", "open", "terminal"])
def test_old_event_stream_callbacks_cannot_touch_the_new_run(callback) -> None:
    rendered = _run_async_ui(r"""
const {callback} = JSON.parse(process.argv[2]);
call('beginRun', job('old-job'), 'job');
const oldSource = sources[0];
call('beginRun', job('new-job', 2), 'job');
const currentSource = sources[1];
vm.runInContext('state.sseFailures = 2', context);
if (callback === 'message') oldSource.onmessage({data: JSON.stringify(job('old-job', 40, 'complete', '633'))});
if (callback === 'malformed') oldSource.onmessage({data: 'invalid JSON'});
if (callback === 'error') {oldSource.onerror(); oldSource.onerror(); oldSource.onerror();}
if (callback === 'open') oldSource.onopen();
if (callback === 'terminal') await oldSource.listeners.terminal();
process.stdout.write(JSON.stringify({id: readState().latestSnapshot.id, active: !currentSource.closed,
  notice: get('job-status').textContent, failures: readState().sseFailures,
  polling: readState().pollTimer, requests}));
""", {"callback": callback})
    assert rendered["id"] == "new-job"
    assert rendered["active"]
    assert rendered["notice"] == ""
    assert rendered["failures"] == 2
    assert rendered["polling"] is None
    assert rendered["requests"] == []


@pytest.mark.parametrize("outcome", ["success", "error"])
def test_old_cancellation_response_cannot_change_the_new_run(outcome) -> None:
    rendered = _run_async_ui(r"""
const {outcome} = JSON.parse(process.argv[2]);
const pending = deferred();
context.fetch = url => {requests.push(url); return pending.promise;};
call('beginRun', job('old-job'), 'job');
const cancellation = call('cancelCurrent');
call('beginRun', job('new-job', 2), 'job');
pending.resolve(outcome === 'success' ? response({}) : {ok: false, json: async () => ({detail: 'Old cancellation failed'})});
await cancellation;
process.stdout.write(JSON.stringify({progress: get('progress-status').textContent,
  notice: get('job-status').textContent, cancelDisabled: get('cancel-button').disabled, requests}));
""", {"outcome": outcome})
    assert rendered["progress"] == "new-job"
    assert rendered["notice"] == ""
    assert rendered["cancelDisabled"] is False
    assert rendered["requests"] == ["/api/jobs/old-job/cancel"]


@pytest.mark.parametrize("action", ["copyExport", "downloadExport"])
@pytest.mark.parametrize("outcome", ["success", "error"])
def test_delayed_export_cannot_publish_old_bytes_under_a_new_run(action, outcome) -> None:
    rendered = _run_async_ui(r"""
const {action, outcome} = JSON.parse(process.argv[2]);
const pending = deferred(), copied = [], downloads = [];
Object.assign(context, {TextDecoder, Uint8Array,
  navigator: {clipboard: {writeText: async text => copied.push(text)}},
  Blob: class {}, URL: {createObjectURL: () => 'blob:local', revokeObjectURL: () => {}},
  fetch: url => {requests.push(url); return pending.promise;},
});
Element.prototype.click = function() {downloads.push(this.download);};
call('beginRun', job('old-job', 40, 'complete', '633'), 'job');
const exporting = call(action);
call('beginRun', job('new-job', 2, 'complete', '255'), 'job');
get('copy-status').textContent = '';
const bytes = new TextEncoder().encode('{"cat":["633"]}');
pending.resolve(outcome === 'success' ? {ok: true, arrayBuffer: async () => bytes.buffer}
  : {ok: false, json: async () => ({detail: 'Old export failed'})});
await exporting;
process.stdout.write(JSON.stringify({copied, downloads, notice: get('copy-status').textContent,
  summary: get('summary-content').textContent, requests}));
""", {"action": action, "outcome": outcome})
    assert rendered["copied"] == [] and rendered["downloads"] == []
    assert rendered["notice"] == ""
    assert "Category 255" in rendered["summary"]
    assert rendered["requests"] == ["/api/jobs/old-job/export"]


@pytest.mark.parametrize("mode", ["path", "upload"])
@pytest.mark.parametrize("change", ["source", "mode"])
@pytest.mark.parametrize("outcome", ["success", "error"])
def test_changed_selection_ignores_pending_inspection_and_keeps_inputs_usable(mode, change, outcome) -> None:
    rendered = _run_async_ui(r"""
const {mode, change, outcome} = JSON.parse(process.argv[2]);
const pending = deferred();
context.fetch = () => pending.promise;
call('showMode', mode);
get('source-path').value = '/media/first.wav';
get('source-file').files = [{name: 'first.wav'}];
const inspection = call('handleSubmit', {preventDefault() {}});
get('source-path').value = '/media/second.wav';
get('source-file').files = [{name: 'second.wav'}];
if (change === 'mode') call('showMode', mode === 'path' ? 'upload' : 'path');
else call('resetPreparedSource');
get('progress-status').textContent = 'Current selection';
const usable = !get('source-path').disabled && !get('source-file').disabled;
pending.resolve(outcome === 'error' ? {ok: false, json: async () => ({detail: 'Old source failed'})}
  : response(mode === 'upload' ? {upload_id: 'old-upload', probe} : probe));
await inspection;
process.stdout.write(JSON.stringify({prepared: readState().prepared, usable,
  trackHidden: get('track-options').hidden, button: get('analyze-button').textContent,
  buttonDisabled: get('analyze-button').disabled, error: get('form-error').textContent,
  progress: get('progress-status').textContent}));
""", {"mode": mode, "change": change, "outcome": outcome})
    assert rendered["prepared"] is None
    assert rendered["usable"]
    assert rendered["trackHidden"]
    assert rendered["button"] == "Check audio track"
    assert rendered["buttonDisabled"] is False
    assert rendered["error"] == ""
    assert rendered["progress"] == "Current selection"


@pytest.mark.parametrize("mode", ["path", "upload", "batch"])
@pytest.mark.parametrize("outcome", ["success", "error"])
def test_delayed_submission_cannot_select_its_run_after_a_new_run_starts(mode, outcome) -> None:
    rendered = _run_async_ui(r"""
const {mode, outcome} = JSON.parse(process.argv[2]);
const pending = deferred();
context.fetch = () => pending.promise;
call('showMode', mode);
get('source-directory').value = '/media';
get('audio-stream').value = '0';
get('content-context').value = '';
vm.runInContext('state.prepared = {mode: ' + JSON.stringify(mode) + ', path: "/media/old.wav", uploadId: "old-upload"}', context);
const submission = call('handleSubmit', {preventDefault() {}});
call('beginRun', job('new-job', 2), 'job');
pending.resolve(outcome === 'success' ? response(mode === 'batch' ? batch('old-batch') : job('old-job'))
  : {ok: false, json: async () => ({detail: 'Old submission failed'})});
await submission;
process.stdout.write(JSON.stringify({id: readState().latestSnapshot.id, kind: readState().itemKind,
  progress: get('progress-status').textContent, error: get('form-error').textContent}));
""", {"mode": mode, "outcome": outcome})
    assert rendered["id"] == "new-job" and rendered["kind"] == "job"
    assert rendered["progress"] == "new-job"
    assert rendered["error"] == ""


@pytest.mark.parametrize("mode", ["path", "upload"])
def test_old_inspection_cannot_unlock_or_replace_a_new_pending_inspection(mode) -> None:
    rendered = _run_async_ui(r"""
const {mode} = JSON.parse(process.argv[2]);
const pending = [deferred(), deferred()];
let requestIndex = 0;
context.fetch = () => pending[requestIndex++].promise;
call('showMode', mode);
get('source-path').value = '/media/first.wav';
get('source-file').files = [{name: 'first.wav'}];
const first = call('handleSubmit', {preventDefault() {}});
get('source-path').value = '/media/second.wav';
get('source-file').files = [{name: 'second.wav'}];
call('resetPreparedSource');
const second = call('handleSubmit', {preventDefault() {}});
pending[0].resolve(response(mode === 'upload' ? {upload_id: 'old-upload', probe} : probe));
await first;
const during = {disabled: get('analyze-button').disabled, prepared: readState().prepared};
pending[1].resolve(response(mode === 'upload' ? {upload_id: 'new-upload', probe} : probe));
await second;
process.stdout.write(JSON.stringify({during, prepared: readState().prepared,
  disabled: get('analyze-button').disabled, button: get('analyze-button').textContent}));
""", {"mode": mode})
    assert rendered["during"] == {"disabled": True, "prepared": None}
    expected = {"mode": "path", "path": "/media/second.wav"} if mode == "path" else {
        "mode": "upload", "uploadId": "new-upload", "path": None,
    }
    assert rendered["prepared"] == expected
    assert rendered["disabled"] is False and rendered["button"] == "Start analysis"


@pytest.mark.parametrize("mode", ["path", "upload", "batch"])
def test_editing_next_source_does_not_lose_an_accepted_submission(mode) -> None:
    rendered = _run_async_ui(r"""
const {mode} = JSON.parse(process.argv[2]);
const pending = deferred();
context.fetch = () => pending.promise;
call('showMode', mode);
get('source-path').value = '/media/first.wav';
get('source-directory').value = '/media/first';
get('audio-stream').value = '0';
get('content-context').value = '';
vm.runInContext('state.prepared = {mode: ' + JSON.stringify(mode) + ', path: "/media/first.wav", uploadId: "first-upload"}', context);
const submission = call('handleSubmit', {preventDefault() {}});
get('source-path').value = '/media/next.wav';
get('source-directory').value = '/media/next';
get('source-file').files = [{name: 'next.wav'}];
call('resetPreparedSource');
const disabledWhileSubmitting = get('analyze-button').disabled;
pending.resolve(response(mode === 'batch' ? batch('accepted-run') : job('accepted-run')));
await submission;
process.stdout.write(JSON.stringify({disabledWhileSubmitting, id: readState().itemId,
  kind: readState().itemKind, cancelVisible: !get('cancel-button').hidden,
  prepared: readState().prepared, path: get('source-path').value,
  directory: get('source-directory').value, file: get('source-file').files[0].name,
  buttonDisabled: get('analyze-button').disabled}));
""", {"mode": mode})
    assert rendered["disabledWhileSubmitting"]
    assert rendered["id"] == "accepted-run"
    assert rendered["kind"] == ("batch" if mode == "batch" else "job")
    assert rendered["cancelVisible"]
    assert rendered["prepared"] is None
    assert rendered["path"] == "/media/next.wav"
    assert rendered["directory"] == "/media/next"
    assert rendered["file"] == "next.wav"
    assert not rendered["buttonDisabled"]


def test_restarting_the_same_job_ignores_an_earlier_runs_pending_response() -> None:
    rendered = _run_async_ui(r"""
const pending = deferred();
context.fetch = () => pending.promise;
call('beginRun', job('same-job'), 'job');
const refreshing = call('refreshJob');
call('beginRun', job('same-job', 2), 'job');
pending.resolve(response(job('same-job', 40, 'complete', '633')));
await refreshing;
process.stdout.write(JSON.stringify({revision: readState().latestRevision,
  active: !sources[1].closed, copyHidden: get('copy-button').hidden}));
""")
    assert rendered == {"revision": 2, "active": True, "copyHidden": True}


def test_snapshot_for_another_job_cannot_replace_the_selected_job() -> None:
    rendered = _run_async_ui(r"""
call('beginRun', job('selected-job', 2), 'job');
sources[0].onmessage({data: JSON.stringify(job('other-job', 40, 'complete', '633'))});
process.stdout.write(JSON.stringify({id: readState().latestSnapshot.id,
  active: !sources[0].closed, copyHidden: get('copy-button').hidden}));
""")
    assert rendered == {"id": "selected-job", "active": True, "copyHidden": True}


@pytest.mark.parametrize("kind", ["job", "batch"])
def test_queued_poll_callback_from_an_old_run_does_not_refresh_a_new_run(kind) -> None:
    rendered = _run_async_ui(r"""
const {kind} = JSON.parse(process.argv[2]);
const make = kind === 'job' ? job : batch;
call('beginRun', make('old-run'), kind);
if (kind === 'job') call('startJobPolling');
const oldCallback = timers.get(readState().pollTimer);
call('beginRun', make('new-run'), kind);
oldCallback();
await new Promise(resolve => setImmediate(resolve));
process.stdout.write(JSON.stringify({requests, notice: get('job-status').textContent}));
""", {"kind": kind})
    assert rendered == {"requests": [], "notice": ""}


def test_page_discloses_experimental_transcript_only_before_analysis() -> None:
    _, text = _page()
    assert "experimental transcript-only" in text
    assert "transcription errors" in text
    assert text.index("experimental transcript-only") < text.index("check audio track")


@pytest.mark.parametrize("mode", ["transcript_only", None])
def test_snapshot_discloses_server_mode_and_active_verification_window(mode) -> None:
    rendered = _run_ui(r"""
const snapshot = JSON.parse(process.argv[2]);
vm.runInContext('state.itemKind = "job"', context);
call('renderSnapshot', snapshot);
const get = id => document.getElementById(id).textContent;
process.stdout.write(JSON.stringify({mode: get('evidence-mode'), stage: get('current-stage'),
  unit: get('current-unit'), progress: get('progress-status'), verification: get('verification-content')}));
""", {
        "evidence_mode": mode, "status": "analysing", "verification_windows": [],
        "progress": {"revision": 4, "stage": "analysing", "percent": 70,
                     "current_chunk": 2, "total_chunks": 3,
                     "message": "transcript verification: window 2 of 3 (30.000–60.000s), 0/5 calls completed; starting"},
    })
    if mode:
        assert "transcript_only" in rendered["mode"]
        assert "Experimental" in rendered["mode"]
    else:
        assert "Legacy" in rendered["mode"] and "not recorded" in rendered["mode"]
        assert "Supported" not in rendered["verification"]
    assert rendered["stage"] == "Classifying and verifying transcript"
    assert rendered["unit"] == "Analysis window 2 of 3"
    assert rendered["progress"].endswith("0/5 calls completed; starting")


@pytest.mark.parametrize(("status", "error", "expected"), [
    ("supported", None, "Supported transcript evidence"),
    ("abstained", "INSUFFICIENT_TOPIC_EVIDENCE", "No supported IAB topic"),
    ("technical_failure", "INVALID_VERIFIER_OUTPUT", "Verification technical failure"),
])
def test_verification_view_uses_saved_server_outcomes_citations_and_policy(status, error, expected) -> None:
    rendered = _run_ui(r"""
const snapshot = JSON.parse(process.argv[2]);
vm.runInContext('state.itemKind = "job"', context);
call('renderSnapshot', snapshot);
const view = document.getElementById('verification-content');
const walk = item => [item.tag, ...item.children.flatMap(walk)];
process.stdout.write(JSON.stringify({text: view.textContent, tags: walk(view),
  mode: document.getElementById('evidence-mode').textContent,
  timeline: document.getElementById('timeline-content').textContent}));
""", {
        "evidence_mode": "transcript_only", "status": "analysing",
        "progress": {"revision": 5, "stage": "analysing"},
        "verification_windows": [{
            "status": status, "error_code": error,
            "verification_policy_version": "english-transcript-verifier-v1",
            "time_range": {"start_seconds": 30, "end_seconds": 60},
            "citations": ([{"category_id": category, "segment_indices": [index],
                           "time_ranges": [{"start_seconds": start, "end_seconds": end}]}
                          for category, index, start, end in [("633", 0, 32, 39), ("255", 2, 48, 54)]
                          ] if status == "supported" else []),
        }],
        # A classifier proposal is deliberately present but must never become support.
        "classification_proposal": {"primary_category": "999", "genre": "<script>acoustic</script>"},
    })
    assert expected in rendered["text"]
    assert "0:30–1:00" in rendered["text"]
    assert "english-transcript-verifier-v1" in rendered["text"]
    assert "english-transcript-verifier-v1" in rendered["mode"]
    assert "999" not in rendered["text"] and "acoustic" not in rendered["text"]
    if status == "supported":
        for phrase in ("Primary", "Category 633", "0:32–0:39", "Secondary", "Category 255", "0:48–0:54", "0", "2"):
            assert phrase in rendered["text"]
    else:
        assert "Category" not in rendered["text"]
    if error:
        assert error in rendered["text"]
    assert "script" not in rendered["tags"]


def test_copy_download_and_failed_export_use_server_bytes_and_render_errors_as_text() -> None:
    rendered = _run_ui(r"""
(async () => {
  const bytes = new TextEncoder().encode('{"cat":["633"],"title":"café"}\n');
  let copied = null, downloaded = null, failed = false;
  const requests = [];
  Object.assign(context, {TextDecoder, Uint8Array,
    navigator: {clipboard: {writeText: async text => {copied = text;}}},
    Blob: class {constructor(parts) {downloaded = Array.from(parts[0]);}},
    URL: {createObjectURL: () => 'blob:local', revokeObjectURL: () => {}},
    fetch: async url => {
      requests.push(url);
      return failed ? {ok: false, json: async () => ({detail: '<img src=x> failed export'})}
        : {ok: true, arrayBuffer: async () => bytes.buffer};
    },
  });
  Element.prototype.click = () => {};
  vm.runInContext('state.itemId = "saved-job"; state.itemKind = "job"', context);
  await call('copyExport');
  await call('downloadExport');
  failed = true;
  await call('copyExport');
  process.stdout.write(JSON.stringify({copied, downloaded, requests,
    error: document.getElementById('copy-status').textContent,
    children: document.getElementById('copy-status').children.length}));
})().catch(error => {console.error(error); process.exitCode = 1;});
""")
    expected = '{"cat":["633"],"title":"café"}\n'
    assert rendered["copied"] == expected
    assert rendered["downloaded"] == list(expected.encode("utf-8"))
    assert rendered["requests"] == ["/api/jobs/saved-job/export"] * 3
    assert rendered["error"] == "Copy unavailable: <img src=x> failed export"
    assert rendered["children"] == 0


def test_client_reconnect_keeps_newest_saved_diagnostics_and_cancelled_export_hidden() -> None:
    rendered = _run_ui(r"""
const sources = [];
context.EventSource = class {
  constructor(url) {this.url = url; sources.push(this);}
  close() {this.closed = true;}
  addEventListener() {}
};
vm.runInContext('state.itemKind = "job"; state.itemId = "saved-job"', context);
const snapshot = {id: 'saved-job', evidence_mode: 'transcript_only', status: 'failed',
  error: '<script>broken model</script>', progress: {revision: 7, stage: 'failed'},
  verification_windows: [{status: 'technical_failure', error_code: '<img src=x> invalid output',
    time_range: {start_seconds: 0, end_seconds: 10}, verification_policy_version: 'english-transcript-verifier-v1', citations: []}],
  transcript: [{time_range: {start_seconds: 0, end_seconds: 3}, text: '<img src=x> spoken'}]};
call('connectEvents');
sources[0].onmessage({data: JSON.stringify(snapshot)});
const first = document.getElementById('verification-content').textContent;
call('connectEvents');
sources[1].onmessage({data: JSON.stringify({...snapshot, progress: {revision: 6, stage: 'analysing'}, verification_windows: []})});
const stale = document.getElementById('verification-content').textContent;
call('renderSnapshot', snapshot, true);
const reconnected = document.getElementById('verification-content').textContent;
const failed = document.getElementById('job-status').textContent;
call('renderSnapshot', {...snapshot, status: 'cancelled', error: null, progress: {revision: 8, stage: 'cancelled'}});
call('renderBatchChildren', [{status: 'failed', error: '<img src=x> model failure', request: {source: '/media/<img src=x>.wav'}}]);
const walk = item => [item.tag, ...item.children.flatMap(walk)];
process.stdout.write(JSON.stringify({first, stale, reconnected, failed,
  cancelled: document.getElementById('job-status').textContent,
  transcript: document.getElementById('transcript-content').textContent,
  batch: document.getElementById('batch-outcomes').textContent,
  tags: [...walk(document.getElementById('verification-content')), ...walk(document.getElementById('batch-outcomes'))],
  copyHidden: document.getElementById('copy-button').hidden,
  downloadHidden: document.getElementById('download-button').hidden}));
""")
    assert rendered["first"] == rendered["stale"] == rendered["reconnected"]
    assert "Verification technical failure" in rendered["first"]
    assert "<img src=x> invalid output" in rendered["first"]
    assert "Analysis stopped: <script>broken model</script>" in rendered["failed"]
    assert "Cancelled. Partial evidence" in rendered["cancelled"]
    assert "<img src=x> spoken" in rendered["transcript"]
    assert "<img src=x>.wav" in rendered["batch"]
    assert "<img src=x> model failure" in rendered["batch"]
    assert "img" not in rendered["tags"] and "script" not in rendered["tags"]
    assert rendered["copyHidden"] and rendered["downloadHidden"]


_V3_SCRIPT = _DOM_HARNESS.split("const snapshot = JSON.parse")[0] + r"""
const base = JSON.parse(process.argv[2]);
vm.runInContext('state.itemKind = "job"; state.itemId = "saved-job"', context);
const get = id => document.getElementById(id);
call('renderSnapshot', base, true);
const complete = {summary: get('summary-content').textContent, excluded: get('excluded-ranges').textContent,
  sentiment: get('sentiment-content').textContent};
call('renderSnapshot', {
  ...base, status: 'failed', error: 'TECHNICAL_FAILURE_SHARE_EXCEEDED', ortb_content: null,
  progress: {...base.progress, revision: 9, stage: 'failed'},
});
process.stdout.write(JSON.stringify({complete, failedStatus: get('job-status').textContent}));
"""


def test_v3_result_explains_structure_partial_coverage_and_cap_failures() -> None:
    if not shutil.which("node"):
        pytest.skip("Node.js is required for the browser-script DOM contract")
    segment = lambda key, value: {"id": key, "value": value}  # noqa: E731
    snapshot = {
        "id": "saved-job", "status": "complete", "error": None,
        "progress": {"revision": 8, "stage": "complete", "percent": 100, "message": "Saved"},
        "ortb_content": {"cat": ["1", "391"], "context": 1, "kwarray": [], "data": [{
            "name": "contextual_engine", "segment": [
                segment("topic_structure", "multi_topic"),
                segment("coverage", "partial_technical_failures"),
            ],
        }]},
        "text_timeline": [], "topic_abstentions": [], "transcript": [],
        "coverage": {
            "total_windows": 3, "successful_windows": 2, "transcribed_seconds": 50.0,
            "classified_units": 2, "failed_ranges": [], "abstained_ranges": [],
            "excluded_ranges": [{
                "time_range": {"start_seconds": 30, "end_seconds": 60},
                "error_code": "NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE",
                "stage": "language_detection", "reason": "non_english_or_unconfirmed_language",
            }],
        },
    }
    result = subprocess.run(
        ["node", "-e", _V3_SCRIPT, str(STATIC / "app.js"), json.dumps(snapshot)],
        capture_output=True, text=True, check=True,
    )
    rendered = json.loads(result.stdout)
    assert "Multi-topic: the listed topics are co-equal" in rendered["complete"]["summary"]
    assert "Partial coverage: some speech failed processing" in rendered["complete"]["summary"]
    assert "not confirmed as English" in rendered["complete"]["excluded"]
    assert rendered["failedStatus"].startswith("More than 20% of the English speech failed processing")


def test_declared_music_result_says_the_label_was_not_detected() -> None:
    if not shutil.which("node"):
        pytest.skip("Node.js is required for the browser-script DOM contract")
    snapshot = {
        "id": "saved-job", "status": "complete", "error": None,
        "progress": {"revision": 8, "stage": "complete", "percent": 100, "message": "Saved"},
        "ortb_content": {"cat": ["338", "650"], "context": 3, "data": [{
            "name": "contextual_engine",
            "segment": [{"id": "topic_structure", "value": "declared_music"}],
        }]},
        "sentiment": {"label": "positive", "score": 0.8},
        "text_timeline": [], "topic_abstentions": [], "transcript": [],
        "coverage": {
            "total_windows": 1, "successful_windows": 1, "transcribed_seconds": 10.0,
            "classified_units": 0, "failed_ranges": [], "abstained_ranges": [], "excluded_ranges": [],
        },
    }
    result = subprocess.run(
        ["node", "-e", _V3_SCRIPT, str(STATIC / "app.js"), json.dumps(snapshot)],
        capture_output=True, text=True, check=True,
    )
    rendered = json.loads(result.stdout)["complete"]
    assert "Declared: Music comes from the Music content context you chose" in rendered["summary"]
    assert rendered["sentiment"] == "positiveSigned score 0.8"


def test_upload_retry_stages_the_file_again_instead_of_reusing_a_claimed_id() -> None:
    # Regression: after "not enough topical evidence", choosing Music and
    # starting again sent the already-claimed upload ID.
    rendered = _run_async_ui(r"""
const posts = [];
let uploads = 0;
context.fetch = async (url, options) => {
  if (url.startsWith('/api/uploads')) {
    uploads += 1;
    return response({upload_id: 'upload-' + uploads,
      probe: {audio_streams: [{index: 0, codec_name: 'aac'}], selected_audio_stream: 0}});
  }
  posts.push({url, body: JSON.parse(options.body)});
  return response(job('job-' + posts.length));
};
call('showMode', 'upload');
get('source-file').files = [{name: 'song.mp4'}];
get('content-context').value = '';
await call('handleSubmit', {preventDefault() {}});
get('audio-stream').value = '0';
await call('handleSubmit', {preventDefault() {}});
get('content-context').value = '3';
await call('handleSubmit', {preventDefault() {}});
process.stdout.write(JSON.stringify({posts, uploads, error: get('form-error').textContent}));
""")
    first, retry = rendered["posts"]
    assert first["body"]["upload_id"] == "upload-1"
    assert retry["body"] == {"upload_id": "upload-2", "audio_stream_index": 0, "requested_context": 3}
    assert rendered["uploads"] == 2 and rendered["error"] == ""

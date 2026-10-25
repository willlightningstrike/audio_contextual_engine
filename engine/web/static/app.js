const byId = (id) => document.getElementById(id);
const state = {
  mode: "upload",
  prepared: null,
  selectionGeneration: 0,
  submissionPending: false,
  itemId: null,
  itemKind: null,
  runGeneration: 0,
  latestRevision: -1,
  eventSource: null,
  pollTimer: null,
  sseFailures: 0,
  taxonomy: new Map(),
  taxonomyAvailable: false,
  taxonomyLoaded: false,
  latestSnapshot: null,
  latestBatchChildren: null,
};

const terminalStates = new Set(["complete", "failed", "cancelled"]);
const contextNames = { 1: "Video program", 3: "Music", 6: "Spoken audio", 7: "Unknown" };
const coverageFailures = {
  NON_ENGLISH_SHARE_EXCEEDED: "More than 20% of the speech is not English, so the file is out of scope.",
  TECHNICAL_FAILURE_SHARE_EXCEEDED: "More than 20% of the English speech failed processing, so no result was published.",
  NO_USABLE_ENGLISH_TRANSCRIPT: "No usable English speech was found.",
  TRANSCRIPTION_INCOMPLETE: "The selected audio was not fully processed.",
};
const stageNames = {
  queued: "Queued",
  extracting_audio: "Extracting audio",
  detecting_language: "Checking English eligibility",
  transcribing: "Transcribing",
  analysing: "Classifying and verifying transcript",
  exporting: "Validating result",
  cancelling: "Cancelling",
  batch_processing: "Processing folder",
  complete: "Complete",
  failed: "Failed",
  cancelled: "Cancelled",
};

function node(tag, className = "", text = null) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== null) item.textContent = String(text);
  return item;
}

function errorText(error) {
  return error instanceof Error ? error.message : "The local service could not complete the request.";
}

async function jsonRequest(url, options = {}) {
  const response = await fetch(url, options);
  let payload = null;
  try { payload = await response.json(); } catch { /* error body may not be JSON */ }
  if (!response.ok) {
    const detail = payload?.detail;
    const message = Array.isArray(detail) ? detail.map((item) => item.msg || "Invalid request").join("; ") : detail;
    throw new Error(message || `The local service returned ${response.status}.`);
  }
  return payload;
}

function jsonOptions(body) {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

function selectedMode() {
  return document.querySelector('input[name="source-mode"]:checked')?.value || "upload";
}

function setFormError(message = "") {
  const target = byId("form-error");
  target.textContent = message;
  target.hidden = !message;
}

function resetPreparedSource() {
  state.selectionGeneration += 1;
  state.prepared = null;
  byId("track-options").hidden = true;
  byId("audio-stream").replaceChildren(node("option", "", "Check the source first"));
  byId("audio-stream").disabled = true;
  byId("analyze-button").textContent = "Check audio track";
  byId("analyze-button").disabled = state.submissionPending;
}

function showMode(mode) {
  state.mode = mode;
  document.querySelectorAll("[data-source-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.sourcePanel !== mode;
  });
  resetPreparedSource();
  setFormError();
}

function showProbe(probe, source) {
  const selector = byId("audio-stream");
  selector.replaceChildren();
  for (const stream of probe.audio_streams || []) {
    const option = node("option", "", `Track ${stream.index} · ${stream.codec_name}${stream.language ? ` · ${stream.language}` : ""}`);
    option.value = String(stream.index);
    option.selected = stream.index === probe.selected_audio_stream;
    selector.append(option);
  }
  if (!selector.options.length) throw new Error("No usable audio track was found in this source.");
  selector.disabled = false;
  byId("track-options").hidden = false;
  byId("analyze-button").textContent = "Start analysis";
  state.prepared = source;
}

async function inspectSource() {
  const selection = state.selectionGeneration;
  const run = state.runGeneration;
  if (state.mode === "upload") {
    const file = byId("source-file").files[0];
    if (!file) throw new Error("Choose an audio or video file first.");
    const result = await jsonRequest(`/api/uploads?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      body: file,
    });
    if (selection !== state.selectionGeneration || run !== state.runGeneration) return;
    showProbe(result.probe, { mode: "upload", uploadId: result.upload_id, path: null });
    byId("progress-status").textContent = `${file.name} is ready. Choose an audio track, then start analysis.`;
    return;
  }
  const path = byId("source-path").value.trim();
  if (!path) throw new Error("Enter a full path to a media file.");
  const result = await jsonRequest("/api/media/probe-path", jsonOptions({ path }));
  if (selection !== state.selectionGeneration || run !== state.runGeneration) return;
  showProbe(result, { mode: "path", path });
  byId("progress-status").textContent = "Source checked. Choose an audio track, then start analysis.";
}

function contextValue() {
  const value = byId("content-context").value;
  return value ? Number(value) : null;
}

async function createJob() {
  const run = state.runGeneration;
  const audioStream = Number(byId("audio-stream").value);
  const context = contextValue();
  let snapshot;
  if (state.prepared.mode === "upload") {
    const payload = { upload_id: state.prepared.uploadId, audio_stream_index: audioStream };
    if (context !== null) payload.requested_context = context;
    snapshot = await jsonRequest("/api/jobs/upload", jsonOptions(payload));
  } else {
    const payload = { path: state.prepared.path, audio_stream_index: audioStream };
    if (context !== null) payload.requested_context = context;
    snapshot = await jsonRequest("/api/jobs/path", jsonOptions(payload));
  }
  if (run === state.runGeneration) beginRun(snapshot, "job");
}

async function createBatch() {
  const run = state.runGeneration;
  const path = byId("source-directory").value.trim();
  if (!path) throw new Error("Enter a full path to a folder.");
  const snapshot = await jsonRequest("/api/batches/path", jsonOptions({ path }));
  if (run === state.runGeneration) beginRun(snapshot, "batch");
}

async function handleSubmit(event) {
  event.preventDefault();
  if (byId("analyze-button").disabled) return;
  const selection = state.selectionGeneration;
  const run = state.runGeneration;
  const submitting = state.mode === "batch" || state.prepared !== null;
  if (submitting) state.submissionPending = true;
  setFormError();
  byId("analyze-button").disabled = true;
  try {
    if (state.mode === "batch") await createBatch();
    else if (!state.prepared) await inspectSource();
    else await createJob();
  } catch (error) {
    if (selection !== state.selectionGeneration || run !== state.runGeneration) return;
    setFormError(errorText(error));
    byId("progress-status").textContent = "The source could not be started.";
  } finally {
    if (submitting) state.submissionPending = false;
    if (submitting || selection === state.selectionGeneration) {
      byId("analyze-button").disabled = state.submissionPending;
    }
  }
}

function beginRun(snapshot, kind) {
  stopUpdates();
  state.runGeneration += 1;
  state.itemId = snapshot.id;
  state.itemKind = kind;
  state.latestSnapshot = snapshot;
  state.latestBatchChildren = null;
  state.latestRevision = -1;
  state.sseFailures = 0;
  byId("job-results").hidden = false;
  byId("batch-panel").hidden = kind !== "batch";
  byId("cancel-button").hidden = false;
  byId("cancel-button").disabled = false;
  byId("copy-button").hidden = true;
  byId("download-button").hidden = true;
  byId("job-status").textContent = "";
  byId("copy-status").textContent = "";
  updateTaxonomyNotice();
  byId("summary-content").replaceChildren(node("p", "empty-result", "A whole-file result will appear when analysis is complete."));
  byId("timeline-content").replaceChildren(node("p", "muted", "Time-coded evidence will appear as windows finish."));
  renderSnapshot(snapshot, true);
  if (kind === "job") connectEvents();
  else startBatchPolling();
}

function stopUpdates() {
  if (state.eventSource) state.eventSource.close();
  state.eventSource = null;
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
}

function connectEvents() {
  if (!state.itemId || state.itemKind !== "job") return;
  const run = state.runGeneration;
  const source = new EventSource(`/api/jobs/${encodeURIComponent(state.itemId)}/events`);
  state.eventSource = source;
  const isCurrent = () => run === state.runGeneration && state.eventSource === source;
  source.onopen = () => {
    if (!isCurrent()) return;
    state.sseFailures = 0;
  };
  source.onmessage = (event) => {
    if (!isCurrent()) return;
    try {
      renderSnapshot(JSON.parse(event.data));
      state.sseFailures = 0;
    } catch {
      setConnectionNotice("A progress update could not be read. Waiting for the next saved snapshot.");
    }
  };
  source.addEventListener("terminal", async () => {
    if (!isCurrent()) return;
    source.close();
    state.eventSource = null;
    await refreshJob();
  });
  source.onerror = () => {
    if (!isCurrent()) return;
    state.sseFailures += 1;
    if (state.sseFailures >= 3) {
      source.close();
      state.eventSource = null;
      setConnectionNotice("Live updates are unavailable. Checking the saved job every five seconds.");
      startJobPolling();
    }
  };
}

function setConnectionNotice(message) {
  byId("job-status").textContent = message;
}

function startJobPolling() {
  if (state.pollTimer) return;
  const run = state.runGeneration;
  state.pollTimer = setInterval(() => {
    if (run === state.runGeneration) refreshJob();
  }, 5000);
}

function startBatchPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  const run = state.runGeneration;
  state.pollTimer = setInterval(() => {
    if (run === state.runGeneration) refreshBatch();
  }, 5000);
}

async function refreshJob() {
  if (!state.itemId || state.itemKind !== "job") return;
  const run = state.runGeneration;
  try {
    const snapshot = await jsonRequest(`/api/jobs/${encodeURIComponent(state.itemId)}`);
    if (run !== state.runGeneration) return;
    renderSnapshot(snapshot, true);
  } catch (error) {
    if (run !== state.runGeneration) return;
    setConnectionNotice(`Could not refresh the saved job: ${errorText(error)}`);
  }
}

async function refreshBatch() {
  if (!state.itemId || state.itemKind !== "batch") return;
  const run = state.runGeneration;
  try {
    const snapshot = await jsonRequest(`/api/batches/${encodeURIComponent(state.itemId)}`);
    if (run !== state.runGeneration || !renderSnapshot(snapshot, true)) return;
    const children = await Promise.all((snapshot.child_job_ids || []).map(async (id) => {
      try { return await jsonRequest(`/api/jobs/${encodeURIComponent(id)}`); }
      catch { return { id, status: snapshot.child_outcomes?.[id] || "queued", error: "Job snapshot is unavailable." }; }
    }));
    // A newer batch snapshot may have started fetching its own child results.
    if (run !== state.runGeneration || state.latestSnapshot !== snapshot) return;
    renderBatchChildren(children);
    if (terminalStates.has(snapshot.status)) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  } catch (error) {
    if (run !== state.runGeneration) return;
    setConnectionNotice(`Could not refresh the folder run: ${errorText(error)}`);
  }
}

function renderSnapshot(snapshot, allowEqualRevision = false) {
  if (state.itemId && snapshot.id !== state.itemId) return false;
  const revision = Number(snapshot.progress?.revision ?? 0);
  if (revision < state.latestRevision) return false;
  if (revision === state.latestRevision && !allowEqualRevision) return false;
  state.latestRevision = revision;
  state.latestSnapshot = snapshot;
  const progress = snapshot.progress || {};
  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
  byId("progress-bar").style.width = `${percent}%`;
  byId("progress-track").setAttribute("aria-valuenow", String(Math.round(percent)));
  byId("progress-percent").textContent = `${Math.round(percent)}%`;
  byId("current-stage").textContent = stageNames[progress.stage] || "Processing";
  byId("progress-status").textContent = progress.message || snapshot.status || "Waiting";
  const position = progress.current_chunk && progress.total_chunks
    ? `${progress.stage === "analysing" ? "Analysis window" : "Audio unit"} ${progress.current_chunk} of ${progress.total_chunks}`
    : "";
  const file = progress.current_file ? `Current clip: ${progress.current_file}` : "";
  byId("current-unit").textContent = [file, position].filter(Boolean).join(" · ");
  renderEvidenceMode(snapshot);
  renderResults(snapshot);
  renderCoverage(snapshot.coverage);
  updateTerminal(snapshot);
  return true;
}

function updateTerminal(snapshot) {
  const terminal = terminalStates.has(snapshot.status);
  byId("cancel-button").hidden = terminal;
  byId("cancel-button").disabled = false;
  const complete = snapshot.status === "complete" && state.itemKind === "job";
  byId("copy-button").hidden = !complete;
  byId("download-button").hidden = !complete;
  if (snapshot.status === "cancelling") {
    byId("job-status").textContent = "Cancelling. The current bounded step is finishing; saved partial evidence will remain available.";
  } else if (snapshot.status === "cancelled") {
    byId("job-status").textContent = "Cancelled. Partial evidence is shown below; export is unavailable.";
  } else if (snapshot.status === "failed" && snapshot.error === "INSUFFICIENT_TOPIC_EVIDENCE") {
    byId("job-status").textContent = "Not enough topical evidence in the audio. Songs and music videos usually end here, because music is not detected from audio; set Content context to Music to label them.";
  } else if (snapshot.status === "failed" && coverageFailures[snapshot.error]) {
    byId("job-status").textContent = `${coverageFailures[snapshot.error]} Saved partial evidence is shown below; export is unavailable.`;
  } else if (snapshot.status === "failed") {
    byId("job-status").textContent = `Analysis stopped: ${snapshot.error || "processing failed"}. Saved partial evidence is shown below; export is unavailable.`;
  } else if (complete) {
    byId("job-status").textContent = "Analysis complete. The download and copy actions use the same validated JSON bytes.";
  }
  if (terminal && state.itemKind === "job") stopUpdates();
  if (terminal && state.itemKind === "batch" && state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function category(id) {
  const value = state.taxonomy.get(String(id));
  return value ? { name: value.name, path: (value.path || []).join(" › ") } : { name: `Category ${id}`, path: "" };
}

function appendCategory(parent, id, primary = false) {
  const info = category(id);
  if (primary) {
    parent.append(node("h3", "primary-label", info.name));
    if (info.path) parent.append(node("p", "category-path", info.path));
  } else {
    parent.append(node("span", "category-pill", info.name));
  }
}

function segmentsFrom(content) {
  return (content?.data || []).flatMap((group) => group.segment || []);
}

function renderSummary(snapshot) {
  const content = snapshot.ortb_content;
  const container = byId("summary-content");
  const secondaryContainer = byId("secondary-content");
  container.replaceChildren();
  secondaryContainer.replaceChildren();
  if (!content?.cat?.length) {
    const message = snapshot.status === "complete"
      ? "No category result was included in this completed record."
      : "No whole-file result yet. Any completed time-coded classifications remain available below.";
    container.append(node("p", "empty-result", message));
    byId("context-label").textContent = "";
    return;
  }
  appendCategory(container, content.cat[0], true);
  const segments = Object.fromEntries(segmentsFrom(content).map((item) => [item.id, item.value]));
  if (segments.topic_structure === "multi_topic") {
    container.append(node("p", "category-note", "Multi-topic: the listed topics are co-equal; none covers most of the speech on its own."));
  } else if (segments.topic_structure === "parent_rollup") {
    container.append(node("p", "category-note", "Broader topic: several related subtopics together cover most of the speech."));
  } else if (segments.topic_structure === "declared_music") {
    container.append(node("p", "category-note", "Declared: Music comes from the Music content context you chose, not from the audio. Any other topic listed was verified from the transcript."));
  }
  if (segments.coverage === "partial_technical_failures") {
    container.append(node("p", "category-note", "Partial coverage: some speech failed processing and was left out of this result. See Coverage below."));
  }
  if (content.cat.length > 1) {
    const secondaries = node("div", "secondary-block");
    for (const id of content.cat.slice(1, 3)) appendCategory(secondaries, id);
    secondaryContainer.append(secondaries);
  } else {
    secondaryContainer.append(node("p", "category-path", "No secondary topics were sufficiently distinct and substantial."));
  }
  byId("context-label").textContent = contextNames[content.context] || "Unknown context";
}

function renderDetails(snapshot) {
  const content = snapshot.ortb_content;
  const keywords = byId("keywords-content");
  keywords.replaceChildren();
  const words = content?.kwarray || [];
  if (words.length) {
    const row = node("div", "keyword-list");
    words.forEach((word) => row.append(node("span", "keyword-tag", word)));
    keywords.append(row);
  } else keywords.append(node("p", "muted", "No keywords available."));

  const entities = byId("entities-content");
  entities.replaceChildren();
  const entitySegments = segmentsFrom(content).filter((item) => item.id?.startsWith("entity_"));
  if (entitySegments.length) {
    const list = node("div", "entity-list");
    entitySegments.forEach((item) => {
      const row = node("div", "entity-row");
      row.append(node("span", "", item.value));
      row.append(node("span", "entity-type", item.id.slice(7).replaceAll("_", " ")));
      list.append(row);
    });
    entities.append(list);
  } else entities.append(node("p", "muted", "No named entities available."));

  const sentiment = segmentsFrom(content).find((item) => item.id === "sentiment");
  const sentimentTarget = byId("sentiment-content");
  sentimentTarget.replaceChildren();
  if (sentiment) {
    const [label, score] = String(sentiment.value).split(":", 2);
    sentimentTarget.append(node("p", "sentiment-value", label || "Not available"));
    if (score !== undefined) sentimentTarget.append(node("p", "sentiment-score", `Signed score ${score}`));
  } else sentimentTarget.append(node("p", "muted", "No sentiment available."));
}

function rangeText(range) {
  if (!range) return "Unknown time";
  return `${formatTime(range.start_seconds)}–${formatTime(range.end_seconds)}`;
}

function formatTime(seconds) {
  const safe = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remainder = safe % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}` : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function renderEvidenceMode(snapshot) {
  const target = byId("evidence-mode");
  if (state.itemKind === "batch") {
    target.textContent = "Folder run · evidence mode and verification policy are recorded per clip.";
    return;
  }
  const mode = snapshot.evidence_mode;
  const policies = [...new Set((snapshot.verification_windows || [])
    .map((entry) => entry.verification_policy_version).filter(Boolean))];
  const description = mode === "transcript_only"
    ? "Experimental · server evidence mode: transcript_only. Topics depend on transcript evidence and may be wrong."
    : mode ? `Server evidence mode: ${mode}.` : "Legacy record · evidence mode not recorded; transcript-only verification is not established.";
  target.textContent = `${description} ${policies.length ? `Verification policy: ${policies.join(", ")}.` : "Verification policy: no saved window outcome yet."}`;
}

function renderVerification(snapshot) {
  const container = byId("verification-content");
  container.replaceChildren();
  const windows = snapshot.verification_windows || [];
  if (!windows.length) {
    container.append(node("p", "muted", "No saved verification outcomes. Current verification activity appears in analysis progress."));
  }
  const labels = {
    supported: "Supported transcript evidence",
    abstained: "No supported IAB topic",
    technical_failure: "Verification technical failure",
  };
  for (const entry of windows) {
    const article = node("article", "timeline-item");
    article.append(node("time", "timeline-time", rangeText(entry.time_range)));
    const detail = node("div", "timeline-detail");
    detail.append(node("h3", "", labels[entry.status] || "Verification outcome unavailable"));
    detail.append(node("p", "category-path", `Verification policy: ${entry.verification_policy_version || "not recorded"}`));
    if (entry.error_code) detail.append(node("p", "timeline-evidence", entry.error_code));
    // Status and citations are server-derived; classifier proposals are never consulted.
    if (entry.status === "supported") {
      for (const [index, citation] of (entry.citations || []).entries()) {
        detail.append(node("p", "timeline-secondary", `${index === 0 ? "Primary" : "Secondary"}: ${category(citation.category_id).name}`));
        detail.append(node("p", "timeline-evidence", `Transcript segment indices (window-local, zero-based): ${(citation.segment_indices || []).join(", ")} · Citations: ${(citation.time_ranges || []).map(rangeText).join(", ")}`));
      }
    }
    article.append(detail);
    container.append(article);
  }
}

function renderTimeline(snapshot) {
  const container = byId("timeline-content");
  container.replaceChildren();
  const timelineItems = [
    ...(snapshot.text_timeline || []).map((entry) => ({ kind: "category", entry })),
    ...(snapshot.topic_abstentions || []).map((entry) => ({ kind: "abstention", entry })),
  ].sort((a, b) => a.entry.time_range.start_seconds - b.entry.time_range.start_seconds);
  byId("timeline-count").textContent = timelineItems.length ? `${timelineItems.length} ${timelineItems.length === 1 ? "window" : "windows"}` : "";
  if (!timelineItems.length) container.append(node("p", "muted", "Time-coded evidence will appear as windows finish."));
  for (const item of timelineItems) {
    const entry = item.entry;
    const article = node("article", "timeline-item");
    article.append(node("time", "timeline-time", rangeText(entry.time_range)));
    if (item.kind === "abstention") {
      article.append(node("p", "timeline-detail", "No supported IAB topic"));
      container.append(article);
      continue;
    }
    const detail = node("div", "timeline-detail");
    appendCategory(detail, entry.payload.primary_category, true);
    const secondary = entry.payload.secondary_categories || [];
    if (secondary.length) {
      const line = node("p", "timeline-secondary", `Also: ${secondary.map((id) => category(id).name).join(", ")}`);
      detail.append(line);
    }
    const evidence = entry.evidence_ranges || [];
    if (evidence.length) detail.append(node("p", "timeline-evidence", `${evidence.length} transcript ${evidence.length === 1 ? "segment" : "segments"} supported this window.`));
    article.append(detail);
    container.append(article);
  }

  const transcript = byId("transcript-content");
  transcript.replaceChildren();
  for (const segment of snapshot.transcript || []) {
    const item = node("li", "transcript-item");
    item.append(node("time", "timeline-time", rangeText(segment.time_range)));
    item.append(node("span", "", segment.text));
    transcript.append(item);
  }
  byId("transcript-details").hidden = !(snapshot.transcript || []).length;
}

function renderCoverage(coverage) {
  const summary = byId("coverage-summary");
  const failed = byId("failed-ranges");
  const excluded = byId("excluded-ranges");
  const abstained = byId("abstained-ranges");
  failed.replaceChildren();
  excluded.replaceChildren();
  abstained.replaceChildren();
  if (!coverage || !coverage.total_windows) {
    summary.textContent = "Audio-window outcomes will appear as processing continues.";
    return;
  }
  summary.textContent = `${coverage.total_windows} audio windows recorded · ${coverage.successful_windows} succeeded · ${coverage.transcribed_seconds.toFixed(1)} seconds of transcript · ${coverage.classified_units} classified windows`;
  for (const item of coverage.failed_ranges || []) {
    const row = node("div", "range-row");
    row.append(node("span", "range-time", rangeText(item.time_range)));
    row.append(node("span", "", `${item.error_code || "Processing error"}${item.stage ? ` · ${item.stage}` : ""}`));
    failed.append(row);
  }
  for (const item of coverage.excluded_ranges || []) {
    const row = node("div", "range-row excluded");
    row.append(node("span", "range-time", rangeText(item.time_range)));
    row.append(node("span", "", item.error_code === "NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE"
      ? "Speech was not confirmed as English; this range was excluded."
      : "Vocal content was uncertain; this range was excluded."));
    excluded.append(row);
  }
  for (const item of coverage.abstained_ranges || []) {
    const row = node("div", "range-row abstained");
    row.append(node("span", "range-time", rangeText(item.time_range)));
    row.append(node("span", "", "No supported IAB topic"));
    abstained.append(row);
  }
}

function renderResults(snapshot) {
  if (state.itemKind === "batch") return;
  renderSummary(snapshot);
  renderDetails(snapshot);
  renderTimeline(snapshot);
  renderVerification(snapshot);
}

function renderBatchChildren(children) {
  state.latestBatchChildren = children;
  const container = byId("batch-outcomes");
  container.replaceChildren();
  byId("batch-count").textContent = `${children.length} ${children.length === 1 ? "clip" : "clips"}`;
  for (const child of children) {
    const row = node("article", "batch-row");
    const source = child.request?.source || "Media file";
    const filename = String(source).split(/[\\/]/).pop() || "Media file";
    row.append(node("span", "batch-filename", filename));
    row.append(node("span", "batch-state", child.status || "unknown"));
    const content = child.ortb_content;
    const names = content?.cat?.map((id) => category(id).name) || [];
    const insufficient = child.status === "failed" && child.error === "INSUFFICIENT_TOPIC_EVIDENCE";
    row.append(node("span", "batch-labels", names.length ? names.join(" · ") : insufficient ? "No supported IAB topic" : "No completed topic result"));
    if (child.error) row.append(node("span", "batch-error", insufficient ? "Not enough topical evidence in the audio" : child.error));
    container.append(row);
  }
}

async function cancelCurrent() {
  if (!state.itemId || !state.itemKind) return;
  const run = state.runGeneration;
  const button = byId("cancel-button");
  button.disabled = true;
  byId("job-status").textContent = "Cancellation requested. Waiting for the current bounded step to stop.";
  try {
    const collection = state.itemKind === "batch" ? "batches" : "jobs";
    await jsonRequest(`/api/${collection}/${encodeURIComponent(state.itemId)}/cancel`, { method: "POST" });
    if (run !== state.runGeneration) return;
    byId("progress-status").textContent = "Cancelling";
    if (state.itemKind === "job") await refreshJob();
    else await refreshBatch();
  } catch (error) {
    if (run !== state.runGeneration) return;
    button.disabled = false;
    byId("job-status").textContent = `Cancellation was not accepted: ${errorText(error)}`;
  }
}

async function exportBytes() {
  if (!state.itemId) throw new Error("There is no completed job to export.");
  const response = await fetch(`/api/jobs/${encodeURIComponent(state.itemId)}/export`);
  if (!response.ok) {
    let detail = "The validated export is not available.";
    try { detail = (await response.json()).detail || detail; } catch { /* keep default */ }
    throw new Error(detail);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function copyExport() {
  const run = state.runGeneration;
  try {
    const bytes = await exportBytes();
    if (run !== state.runGeneration) return;
    await navigator.clipboard.writeText(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    if (run !== state.runGeneration) return;
    byId("copy-status").textContent = "Copied validated JSON.";
  } catch (error) {
    if (run !== state.runGeneration) return;
    byId("copy-status").textContent = `Copy unavailable: ${errorText(error)}`;
  }
}

async function downloadExport() {
  const run = state.runGeneration;
  const itemId = state.itemId;
  try {
    const bytes = await exportBytes();
    if (run !== state.runGeneration) return;
    const url = URL.createObjectURL(new Blob([bytes], { type: "application/json" }));
    const link = node("a");
    link.href = url;
    link.download = `contextual-engine-${itemId}.json`;
    link.click();
    URL.revokeObjectURL(url);
    byId("copy-status").textContent = "Downloaded validated JSON.";
  } catch (error) {
    if (run !== state.runGeneration) return;
    byId("copy-status").textContent = `Download unavailable: ${errorText(error)}`;
  }
}

async function loadTaxonomy() {
  try {
    const rows = await jsonRequest("/api/taxonomy");
    if (!Array.isArray(rows) || !rows.length || rows.some((row) => !row?.id || !row?.name || !Array.isArray(row.path))) {
      throw new Error("The local taxonomy response is incomplete.");
    }
    state.taxonomy = new Map(rows.map((row) => [String(row.id), row]));
    state.taxonomyAvailable = true;
  } catch {
    state.taxonomy = new Map();
    state.taxonomyAvailable = false;
  }
  state.taxonomyLoaded = true;
  updateTaxonomyNotice();
  if (state.latestSnapshot && state.itemKind === "job") renderResults(state.latestSnapshot);
  if (state.latestBatchChildren) renderBatchChildren(state.latestBatchChildren);
}

function updateTaxonomyNotice() {
  const notice = byId("taxonomy-notice");
  if (!state.taxonomyLoaded) {
    notice.hidden = false;
    notice.textContent = "Loading category names from the local taxonomy…";
  } else if (!state.taxonomyAvailable) {
    notice.hidden = false;
    notice.textContent = "Category names are unavailable; showing taxonomy IDs.";
  } else {
    notice.hidden = true;
    notice.textContent = "";
  }
}

document.querySelectorAll('input[name="source-mode"]').forEach((input) => {
  input.addEventListener("change", () => showMode(selectedMode()));
});
document.querySelectorAll("#source-file, #source-path, #source-directory").forEach((input) => {
  input.addEventListener("input", resetPreparedSource);
  input.addEventListener("change", resetPreparedSource);
});
byId("analysis-form").addEventListener("submit", handleSubmit);
byId("cancel-button").addEventListener("click", cancelCurrent);
byId("copy-button").addEventListener("click", copyExport);
byId("download-button").addEventListener("click", downloadExport);
byId("content-context").addEventListener("change", () => { byId("copy-status").textContent = ""; });
loadTaxonomy();

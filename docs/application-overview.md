# Contextual Engine: application and distribution overview

Updated 2026-10-11. Version 0.1.0 is an experimental local application for a single user on an Apple Silicon Mac with at least 64 GiB of unified memory.

## Purpose and requirements

The engine extracts contextual metadata from English speech or usable English lyrics. It accepts audio files and video containers, including TV programs, podcasts, tutorials, reviews, and music videos. Every contextual decision uses the selected audio track and its transcript. Video-frame analysis is reserved for Phase 2.

A successful result has one primary IAB Content Taxonomy 3.1 category and up to two supported secondary categories. Insufficient evidence can produce no label. Accuracy and inspectable evidence take priority over speed. Longer recordings are processed through the end of the selected track.

| Requirement | Current support |
| --- | --- |
| Platform | Apple Silicon, native arm64, macOS 14 or later |
| Memory | At least 64 GiB physical unified memory; verified on one 64 GiB M1 Max |
| Python | 3.11–3.13; exact package versions in `uv.lock`; root migration tested with 3.13.2 |
| Tools | `uv`, `ffmpeg`, and `ffprobe` on `PATH` |
| Storage | About 23 GiB of model assets, plus environment, cache, jobs, uploads, and temporary audio; allow at least 40 GB free during setup |
| Network | First-time dependency/model acquisition and optional YAMNet preparation; inference is local and offline |
| Interface | Browser on the same Mac, loopback HTTP at `127.0.0.1:8765` |

Startup checks the host, binaries, package pins, forbidden frameworks, writable storage, model manifests and hashes, taxonomy provenance, and offline configuration. Missing or changed assets prevent launch.

## Features

- Analyze `.wav`, `.mp4`, `.mov`, `.m4a`, `.mp3`, `.mkv`, and `.webm` inputs, with explicit audio-stream selection when necessary.
- Submit a local path, browser upload, or folder batch. Uploads default to a 1 GiB limit; local-path inputs avoid that upload limit.
- Screen speech/singing eligibility, detect language, and transcribe eligible English audio with timestamps.
- Retrieve IAB candidates, classify transcript windows, and verify category support against cited transcript segments.
- Aggregate a primary category and up to two optional secondaries, keywords, named entities, sentiment, evidence timelines, and OpenRTB-compatible fields (`cattax=9`).
- Display saved stage/window progress, completed verifier calls, partial transcripts, range diagnostics, success, insufficient evidence, technical failure, and cancellation.
- Cancel at cooperative checkpoints and recover saved state. SSE delivers revisions; the browser falls back to polling after repeated live-update failures.
- Copy/download identical server-validated JSON bytes after successful completion. Unsuccessful jobs retain diagnostics and cannot publish a successful export.

Keyword, entity, sentiment, and other model scores are not calibrated accuracy probabilities. The category verifier checks category citations; it does not independently verify every auxiliary field.

## Processing design

```text
Path / upload / folder batch
  → ffprobe: inspect container and select audio stream
  → ffmpeg: normalize the selected audio track
  → YAMNet MLX: speech/singing eligibility
  → Whisper MLX: language and timestamped transcript
  → Qwen3 embeddings: retrieve IAB candidates
  → Qwen3.6: strict transcript classification and metadata
  → Qwen3.6 verifier: category citations and bounded alternatives
  → completion and whole-file topic-sufficiency gates
  → deterministic aggregation and validated JSON export
```

The pipeline visits the whole selected track in sequential windows. A 30-minute or one-hour input is not represented by a few sampled opening windows. Vocal, language, transcription, and classification exclusions remain visible: processing the entire track does not mean every second yields usable evidence.

The classifier and verifier use the same Qwen3.6 model in separate calls, so correlated mistakes remain possible. The verifier receives transcript evidence and taxonomy identities, with bounded attempts. Completed attempts are persisted before interpretation. Strict JSON/schema and citation validation reject malformed or incomplete generation.

The active `english-audio-aggregation-v3` policy first applies coverage caps, then decides over usable windows only:

- **Excluded ranges.** Vocal-uncertain windows (music, silence) are excluded anywhere, with no cap. Non-English windows are excluded while they are at most 20% of vocal audio. Technical failures in transcription or classification are excluded while they are at most 20% of English speech; the export then reports `coverage = partial_technical_failures`.
- **Single topic.** A category's unioned cited primary-role support must exceed 50% of usable transcript duration and cover at least 25% of selected-audio duration (the v2 rule).
- **Parent rollup.** Otherwise, the most specific taxonomy parent whose descendants together pass the same thresholds.
- **Multi-topic.** Otherwise, two or three non-redundant categories, each at least 20% of usable speech, that together pass the thresholds. They are reported as co-equal.
- **Secondaries.** For single and rollup results, up to two nonredundant categories ranked on all verified support, including categories that won the primary role in some windows.

The previously approved 70% foreground-music/genre rule belongs to the separate acoustic design; that path is disabled because no acoustic candidate was admitted. Policy thresholds are not measured model confidence. Support counts only verifier-cited segments, which is the main remaining cause of abstention.

ASR recovery quarantines the specifically tested wordless terminal segment that starts beyond audio end; other invalid timestamps fail that window, which then counts against the technical cap.

The verifier (`english-transcript-verifier-v2`) and classifier schemas list valid segment indices explicitly and size the verifier's token budget to the window, which closes the causes of all 31 invalid or incomplete generation ranges in the v2 evaluation. Decoding is greedy, so no automatic retry is used.

## Models and dependencies

| Role | Pinned model | Revision |
| --- | --- | --- |
| ASR | `mlx-community/whisper-large-v3-turbo-asr-fp16` | `624c19c9af5603fa73b83bce14d4aeea96156d18` |
| Embeddings | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | `b5d88f1fe49b50d2ac01b4692ca2d387f14f9c72` |
| Classification/verifier | `mlx-community/Qwen3.6-35B-A3B-4bit` | `38740b847e4cb78f352aba30aa41c76e08e6eb46` |
| Vocal eligibility | Converted YAMNet MLX assets | Converter `yamnet-tf-reference-to-mlx-npz-v1`; TensorFlow Models source `c14bf9ad91962cf189f9f58db2132c06247fcd53` |

Adapters use MLX on Metal. `transformers` supplies adapter/tokenizer infrastructure. PyTorch, torchvision, torchaudio, Sentence Transformers, and TensorFlow are prohibited in the application environment. Some MLX dependencies bring image-related transitive libraries; the application does not use them for frame inference.

TensorFlow is used only by the explicit YAMNet conversion tool in a disposable reference environment. Verified converted assets can be imported without TensorFlow. Historical Gemma, Qwen2-Audio, Qwen3-Omni, SAM-Audio, and MOSS experiments are not required for this release.

YAMNet supplies vocal eligibility only, not contextual categories. This version emits no IAB Music branch/genre or Music Video category. English lyric topics may support ordinary non-music categories.

## Project-root installation and launch

```text
contextual_engine/
├── run.sh                      # project-root launch/check
├── pyproject.toml, uv.lock      # exact environment
├── engine/                     # application, adapters, API and static UI
├── scripts/                    # setup, model import/download and checks
├── tests/                      # deterministic regressions
├── docs/                       # overview, designs and evidence
├── .venv/                      # local environment; excluded from Git
└── data/
    ├── standards/              # versioned IAB data and provenance
    ├── models/                 # weights, manifests and index; excluded
    ├── jobs/                   # durable jobs and exports; excluded
    └── uploads/                # staged uploads; excluded
```

After setup, launch from this folder:

```sh
./run.sh --check
./run.sh
```

Open `http://127.0.0.1:8765` and stop with Ctrl-C. The launcher resolves its project directory, uses that directory's `.venv`, and defaults storage to `data/`. It does not load code, models, environment, or index from `.worktrees`. `CONTEXTUAL_ENGINE_DATA_DIR` overrides storage; direct library use retains its existing Application Support default.

For everyday launch or a fresh GitHub checkout, follow the [step-by-step setup guide](setup-guide.md). Setup creates the root environment, downloads pinned snapshots into `data/models/`, installs separately prepared YAMNet assets, and builds the index.

An existing verified bundle can be imported offline after creating a root environment:

```sh
uv sync --locked
.venv/bin/python -m scripts.import_model_bundle \
  --source-models /absolute/path/to/verified/bundle/models \
  --target-models "$PWD/data/models"
./run.sh --check
```

The importer requires all three production manifests and the index. It verifies source/copied files, writes new absolute paths, and publishes the destination only after verification. An existing destination is never overwritten. On APFS it uses independent cloned files sharing unchanged disk blocks; elsewhere it copies files. Import downloads nothing and creates no links back to a worktree. Old source bundles can remain as historical evidence without being runtime dependencies.

Model manifests contain machine-local absolute paths and stay outside Git. Moving a prepared installation requires regenerating its manifests/setup. A fresh clone does not include another user's environment or weights.

## Storage and local service

Journals retain raw attempts, transcripts, filenames, diagnostics, citations, and completed exports. They can contain sensitive source material and stay outside Git. There is no automatic job-history purge. Unclaimed uploads expire after 24 hours when upload storage is next accessed. Processing cleans temporary normalized audio; path jobs do not copy source media. Models/index remain installed until explicitly removed.

The service serves one trusted local user on loopback. It is not an authenticated remote or multi-tenant service. Runtime performs no cloud inference or model downloads. Offline flags block model/tokenizer fetching; they are not an operating-system network firewall.

## Evaluation and limitations

The non-blind v2 development run attempted all 308 available cases from a 359-ID inventory; 51 inputs were unavailable. Inputs include full tracks, excerpts, aliases, near duplicates, and unknown original extents. These outcomes are not population accuracy.

| Outcome | Cases |
| --- | ---: |
| Final primary label | 96 |
| Insufficient topic evidence | 99 |
| Non-English or unconfirmed language | 61 |
| Uncertain vocal presence | 35 |
| Technical failure | 17 |

Known errors include tiny Tissot watch excerpts labeled Remodeling & Construction and a camera clip with an incorrect secondary. A keyboard reference abstained. A 20-minute editing tutorial processed all windows but abstained when window topics fragmented. A 93-minute baseball program processed the selected track but could not publish a final label because some windows failed verification.

Technical failures include invalid structured output, incomplete generation, and model failure. Whisper errors and uncertain vocals can prevent usable evidence, especially in singing, mixed/low-level sound, or sparse speech. Language gating can exclude mixed or uncertain material. Silence, instrumental music without vocals, other non-vocal sound, and non-English material are outside the contextual-label scope.

The run covered 21.42 hours of selected audio and accumulated 13.17 hours of per-case processing. Labeled cases had median processing time 67.1 seconds and maximum 990.1 seconds; the longest unsuccessful program took 6,526.2 seconds. Peak sampled RSS was 18.3 GiB and MLX memory 19.0 GiB on one 64 GiB M1 Max. These do not establish smaller-machine support or guaranteed speed.

Service/API SSE reconnect and deterministic UI reconnect regression were accepted instead of manual live-browser transport disconnect/reconnect. The manual browser transport event remains unverified. Live cancellation evidence is separate; no cancellation was requested in the 308-case run.

The figures above were produced with `Qwen2.5-32B-Instruct-4bit`. On 2026-10-23 the classifier/verifier switched to `Qwen3.6-35B-A3B-4bit` after a text-stage replay of the 226 saved cases that had analysis windows, using identical transcripts and Qwen3 shortlists under the v3 policy. Qwen3.6 published 115 labels against 108, had no structured-output failures against 4 windows, took a median 18 seconds of text processing per case against 51, and peaked at 20.0 GiB MLX memory against 18.7. Label quality on the 76 disagreements was judged non-blind by a reviewer, not against frozen references. The full 308-case acceptance run has not been repeated with Qwen3.6.

On 2026-10-24 the taxonomy moved from Content Taxonomy 3.0 to 3.1 (`cattax` 7 to 9) and the retriever from `Qwen3-Embedding-0.6B-4bit-DWQ` to `Qwen3-Embedding-4B-4bit-DWQ`. Category IDs are unchanged in 3.1; 44 names or paths changed, including the Music branch moving under Entertainment, so older results may display new names for the same IDs. The embedder switch follows a 226-case text-stage replay on 3.0 with Qwen3.6: 4B published 122 labels against 115, about seven more correct ones with a similar number wrong, and 86% of published window labels were its top-ranked candidate against 64%; peak MLX memory rose from 20.0 to 22.1 GiB. Removing Music categories from the shortlist was also tested and published fewer labels (119), so the shortlist is unfiltered. Neither change has been evaluated on the full acceptance set.

The detailed evaluation records and benchmark manifests are kept by the maintainer and are not part of this repository.

## GitHub distribution and release state

Publish source, static UI, scripts, tests, `pyproject.toml`, `uv.lock`, standards/provenance, the README, and the docs in `docs/`. Git excludes weights, local manifests, environments, caches, worktrees, personal media, uploads, and job/evaluation outputs. Source archives use an explicit allowlist to avoid local models/media; wheels contain application code, UI assets, and standards only.

[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) lists upstream models/data and license metadata. The application source is licensed under the MIT License (`LICENSE`); third-party models and data keep their own terms.

The release remains experimental and needs human review. Future work includes a frozen blind evaluation, reducing false labels and structured-output failures, and Phase 2 frame analysis. Acoustic genres require a separately validated model.

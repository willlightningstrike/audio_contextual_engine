# Contextual Engine: step-by-step setup and run guide

This guide runs the application from the **project folder**. It covers everyday starts and a fresh source installation. The application is an experimental local browser app, not a double-click macOS application bundle.

It analyzes English speech or usable English lyrics from audio files or the selected audio track of a video. It does not analyze video frames or infer musical genre from sound. Instrumental music, other non-vocal sounds, silence, and non-English content are out of scope. Results can be incorrect or withheld when evidence is insufficient; see the [application overview](application-overview.md) for the full limitations.

## A. Everyday start (after installation)

Use this section once Section B has been completed on this Mac. **You do not need to repeat model downloads, YAMNet conversion, or environment installation.**

### 1. Open Terminal and enter the project folder

```sh
cd /path/to/audio_contextual_engine
```

Replace the example with the folder where you installed the project.

### 2. Check the installation

```sh
./run.sh --check
```

Wait for the checks, including local model-file verification, to finish. The success message is:

```text
Contextual Engine setup passed; local assets verified.
```

If the check reports a failure, use [Troubleshooting](#e-troubleshooting) before continuing. Checks do not download replacement models.

### 3. Start the application

```sh
./run.sh
```

Leave this Terminal window open. Startup checks and model/index initialization may take time before the server announces that it is running. There is no virtual-environment activation step.

The launcher uses the project's `.venv` and defaults to its `data/` folder. If you previously set `CONTEXTUAL_ENGINE_DATA_DIR` in your shell, that setting overrides the default. To explicitly use this project's data instead, run:

```sh
CONTEXTUAL_ENGINE_DATA_DIR="$PWD/data" ./run.sh --check
CONTEXTUAL_ENGINE_DATA_DIR="$PWD/data" ./run.sh
```

### 4. Open the browser interface

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) in a browser on the same Mac. Follow [Analyze your first file](#c-analyze-your-first-file) below.

### 5. Stop or restart

In the Terminal running the server, press **Control-C** to stop it. To start again:

```sh
cd /path/to/audio_contextual_engine
./run.sh
```

Closing the browser does not stop the server. Prefer cancelling an active job in the browser and waiting for cancellation before stopping the server.

## B. Install on a new Mac or fresh source checkout

Skip this section if Section A works. Installation requires internet access; normal operation after successful setup is offline-only.

### 1. Confirm the requirements

- Apple Silicon Mac running macOS 14 or newer; Intel Macs, Windows, and Linux are unsupported.
- At least **64 GiB physical unified memory**. Smaller-memory Macs do not pass this version's preflight.
- At least **40 GB free disk space** before setup. Model snapshots alone occupy approximately 19 GiB; the environment, downloads, index, and temporary files need additional space.
- Python 3.11–3.13; the commands below ask `uv` to create a Python 3.13 application environment.
- `uv`, `ffmpeg`, and `ffprobe` available in Terminal.

Check the Mac in Terminal:

```sh
uname -m
sw_vers -productVersion
sysctl -n hw.memsize
df -h .
```

Expect `arm64`, macOS 14 or later, and at least `68719476736` bytes of physical memory. Run Terminal natively, not under Rosetta.

### 2. Install the command-line prerequisites

If Homebrew is not installed, follow the official [Homebrew installation instructions](https://brew.sh/) first, including its printed shell/PATH setup instructions. Then install the missing tools:

```sh
brew install uv ffmpeg
```

These commands follow the official [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) and [FFmpeg Homebrew formula](https://formulae.brew.sh/formula/ffmpeg). FFmpeg includes `ffprobe`. You can skip installation for tools already present.

Verify they are available:

```sh
uv --version
ffmpeg -version
ffprobe -version
```

### 3. Obtain the source and choose its permanent folder

Clone the repository with `git clone` (see the README), or download and extract its GitHub source ZIP.

Place the extracted folder inside your Mac user folder, for example `/Users/yourname/Projects/audio_contextual_engine`. In Terminal, type `cd `, drag that folder from Finder into Terminal, and press Return. Alternatively, replace the example path below with your own:

```sh
cd /Users/yourname/Projects/audio_contextual_engine
pwd
ls run.sh pyproject.toml uv.lock scripts/setup.sh
```

The last command must list all four files. Stay in this folder for all remaining installation commands. Choose the location before setup: virtual environments and model manifests contain absolute paths and cannot simply be moved afterward.

Source distributions do **not** include the Python environment, model weights, your media, or job history. Read [third-party notices](../THIRD_PARTY_NOTICES.md) before redistributing model assets.

### 4. Create the application environment

```sh
uv sync --locked --python 3.13
```

This installs the versions recorded in `uv.lock` into `.venv`. Let it finish successfully before continuing. Do not install PyTorch or TensorFlow into this environment.

### 5. Prepare the pinned YAMNet assets separately

YAMNet helps distinguish vocal and non-vocal audio; it does not provide topic or music-genre labels. Its one-time conversion uses an isolated TensorFlow reference environment. The application itself runs the converted model with MLX.

Run these commands one at a time:

```sh
uv venv --python 3.11 .yamnet-reference
```

```sh
uv pip install --python .yamnet-reference/bin/python tensorflow==2.19.1 tf-keras==2.19.0
```

```sh
.yamnet-reference/bin/python -m scripts.prepare_yamnet \
  --output "$PWD/.yamnet-prepared"
```

Conversion downloads pinned reference inputs, verifies their hashes, and writes `.yamnet-prepared/manifest.json` and its assets. It refuses to overwrite an existing output directory. If a verified `.yamnet-prepared` already exists at this location, reuse it rather than rerunning conversion over it.

If you already have a complete verified production model bundle, you can skip this conversion and the download step below by following [Import an existing model bundle](#d-optional-import-an-existing-model-bundle) instead.

### 6. Download the production models and build the index

```sh
bash scripts/setup.sh "$PWD/.yamnet-prepared"
```

Setup installs/checks the locked application environment, fetches the pinned models, installs the prepared YAMNet assets, builds the local IAB taxonomy index, and runs preflight. Downloads are large; leave Terminal open and wait for successful completion. There is no fixed installation-time estimate.

The production bundle contains MLX Whisper for transcription, Qwen3 embeddings for taxonomy retrieval, Qwen2.5 text classification/verification, and converted YAMNet for audio eligibility. It does not install the historical Qwen2-Audio, Qwen3-Omni, or Gemma candidates.

By default, all installed assets go under the project's `data/models/`. An existing `CONTEXTUAL_ENGINE_DATA_DIR` environment variable overrides that location; use a clean Terminal configuration or intentionally choose the override before setup.

### 7. Check and launch

```sh
./run.sh --check
./run.sh
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). If extraction did not preserve executable permissions, use `bash run.sh --check` and `bash run.sh` instead.

Once setup succeeds, serving uses local assets only and does not download missing files. Future starts need only the launch command, not Steps 4–6 again.

## C. Analyze your first file

### 1. Choose a suitable source

Start with clear English speech, such as a spoken review or podcast. English singing is accepted only when usable lyrics can be transcribed. Supported extensions are `.wav`, `.mp4`, `.mov`, `.m4a`, `.mp3`, `.mkv`, and `.webm`.

Video is only an input container: the engine uses the chosen audio track and never its frames. Instrumental music and sound effects without vocals are not valid test inputs for contextual topics.

### 2. Select the file in the browser

Choose either:

- **Upload a file** → select the **Audio or video file**. The default upload limit is 1 GiB.
- **Use a local file path** → paste its full absolute path in **Full path to a media file**. This avoids uploading/copying the original media and is preferable for large files.

For a local path, paste `/Users/yourname/Media/interview.mp4` without shell quotation marks, backslash escapes, or `file://`. By default, path-based sources must be within your Mac user folder. Copy an external-volume file into that folder or use upload if it fits the limit.

### 3. Check and select the audio track

Click **Check audio track**. Once inspection finishes, select the English **Audio track** if the source has more than one.

Leave **Content context** set to **Automatic** unless you need an explicit metadata override: **Video program**, **Music**, **Spoken audio**, or **Unknown**. An override does not force English detection or enable acoustic genre classification.

### 4. Start and monitor analysis

Click **Start analysis**. Progress shows stages, the current window/range, and saved verification outcomes. Keep the server's Terminal open and the Mac awake while processing.

The pipeline visits the entire selected audio track in sequential windows, including 30-minute or one-hour files. That does not guarantee usable evidence for every second: language, vocal eligibility, transcription, and verification checks can exclude regions or prevent a final result. Longer files take more work; this release has no guaranteed processing-time target.

To stop a job, click **Cancel** and wait for confirmation. Cancellation is cooperative, so the current bounded operation may need to finish first. Completed partial evidence remains available; a cancelled job cannot export a successful result.

### 5. Read the result and save JSON

A successful **Saved topic result** shows one **Primary category** and up to two **Optional secondary categories**, plus transcript-derived metadata and evidence. Review the transcript and timeline before relying on a label.

Click **Copy JSON** or **Download JSON** to obtain the same validated saved result. These actions are available only for a successful completed job.

Other outcomes are intentional and distinct:

- **Not enough topical evidence in the audio**: usable audio/transcript did not support a defensible final topic. No category is forced and no successful export is available.
- **Analysis stopped**: a technical or eligibility failure occurred; inspect the message and partial evidence.
- **Cancelled**: processing stopped at your request; no successful export is available.

### 6. Optional: analyze a folder

Choose **Analyze a folder**, paste its absolute path, and submit the form. Only supported files directly inside that folder are included, not nested subfolders. This mode creates per-clip jobs; inspect each clip's outcome rather than assuming the entire folder succeeded. The current folder-mode submit button retains the text **Check audio track**, but it starts the folder run directly rather than showing the individual-file track-selection step.

## D. Optional: import an existing model bundle

This replaces first-time conversion/model download only when you already have a complete verified production bundle. It does not replace installing the application environment and command-line prerequisites.

From the new project folder, after `uv sync --locked --python 3.13`:

```sh
.venv/bin/python -m scripts.import_model_bundle \
  --source-models /absolute/path/to/verified/bundle/models \
  --target-models "$PWD/data/models"
```

Replace the source example with the real bundle path; quote it if it contains spaces. The source must contain `english-phase1-manifest.json`, `qwen25-manifest.json`, `yamnet-manifest.json`, their referenced assets, and `taxonomy-index/`. Loose model files alone are insufficient.

The destination `data/models` must **not already exist**; import refuses to overwrite it. The importer verifies files, rebases absolute paths, and creates independent local files rather than symlinks to the source. It downloads nothing; environment installation may still need network access.

Then run:

```sh
./run.sh --check
./run.sh
```

## E. Troubleshooting

### `zsh: no such file or directory: .venv/bin/python`

First confirm you are in the folder containing `run.sh` and `pyproject.toml`:

```sh
pwd
ls run.sh pyproject.toml
```

If `.venv` is missing, create it with `uv sync --locked --python 3.13`. This installs Python packages, not models; a new installation still needs Section B or D. Do not copy a virtual environment from `.worktrees` or another Mac.

### `permission denied: ./run.sh`

Use the equivalent commands:

```sh
bash run.sh --check
bash run.sh
```

### `command not found: uv`, `ffmpeg`, or `ffprobe`

Complete Section B, Step 2 and follow the installer's PATH instructions. Open a new Terminal afterward and check the versions again. Both FFmpeg tools must be visible to the server process.

### Setup reports missing assets, bad hashes, or index/manifest errors

Do not bypass preflight, edit checksums, or rename historical manifests. Confirm `CONTEXTUAL_ENGINE_DATA_DIR` is not pointing at an old installation. Repair the bundle through the setup/import procedure and rerun `./run.sh --check`. Preserve the exact diagnostic if repair fails.

### Memory check fails on a Mac that has 64 GiB or more

Run `sysctl -n hw.memsize` in native macOS Terminal. A restricted execution environment can deny that query with `Operation not permitted`; preflight then cannot verify the required memory and reports a memory-requirement failure. Try the documented launch from native Terminal rather than bypassing the check. If the reported physical memory really is below `68719476736` bytes, this version does not support that Mac.

### `No Metal device available`

Try launching from native macOS Terminal on the Apple Silicon Mac, not a restricted/headless execution environment or Rosetta shell. Check `uname -m` reports `arm64`. A small diagnostic from the project folder is:

```sh
.venv/bin/python -c 'import mlx.core as mx; print(mx.array([1]))'
```

An array result confirms that small MLX operation works in that environment; it is not a full application/model test. If it still fails, preserve the full error and host details for investigation. There is no supported CPU-only or PyTorch fallback.

### Browser cannot connect, or port 8765 is already in use

Wait until startup completes and confirm the server Terminal has not exited with an error. Use `http://127.0.0.1:8765`, not an HTTPS or remote-Mac address. If an earlier instance is still running, use it or stop it with Control-C in its own Terminal before starting another. Do not terminate an unidentified process merely to free the port.

### Prepared installation was moved to a different folder

Recreate `.venv` at the new location and regenerate/import the model bundle there. Model manifests contain absolute paths, and simply copying the old environment and `data/models` is insufficient. The importer requires a new, absent destination; do not delete your only model copy to make room for import.

### Valid English audio produces no category

This is possible in the experimental version. Review the transcript, range diagnostics, and verifier outcomes. Incorrect transcription, ambiguous or fragmented topics, insufficient supported duration, or a verification failure can prevent publication. Selecting **Music** or another context cannot force a topic; do not interpret abstention as proof the file contains no meaningful subject.

## F. Where files are stored

With the default project-root launch:

```text
contextual_engine/
├── .venv/                  Application Python environment
├── data/models/            Verified models, manifests, and taxonomy index
├── data/jobs/              Saved job evidence and successful result.json exports
└── data/uploads/           Browser-upload staging
```

The source ZIP/Git repository excludes these generated/local assets. Jobs can contain sensitive transcripts and filenames; they are retained without an automatic history purge. Unclaimed uploads expire after 24 hours when upload storage is next accessed. Path-based processing leaves the original media in place, and temporary normalized audio is cleaned after processing.

Keep `.venv`, the required models, manifests, and index for normal operation. The launcher has no runtime dependency on `.worktrees`, `.yamnet-reference`, or `.yamnet-prepared` once installation has completed and passed checks. Retain any historical evidence or prepared conversion assets you still want before doing separate storage cleanup.

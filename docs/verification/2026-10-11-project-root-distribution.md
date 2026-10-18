# Project-root migration and distribution preparation

Date: 2026-10-11. Application: Contextual Engine 0.1.0, experimental English transcript-only mode.

## Result

The application now launches from the primary project folder using `./run.sh`, its own `.venv`, and `data/models/`. The launcher defaults jobs/uploads to this project's `data/` directory. Runtime code, models, and the taxonomy index do not resolve through `.worktrees`.

The existing root `main` source was retained. The narrow, previously approved ASR terminal-recovery change and its regression tests were brought across from the evaluated worktree, where they had remained uncommitted. Other worktree experiments and unrelated edits were not imported.

The root environment was recreated using Python 3.13.2 and `uv sync --locked --offline --link-mode copy`; the worktree virtual environment was not copied. Its package metadata identifies this project root as the editable source. The managed Python interpreter remains an ordinary external setup prerequisite.

`scripts/import_model_bundle.py` verified the existing production bundle, copied/cloned ASR, embeddings, Qwen2.5, converted YAMNet, and the taxonomy index into `data/models/`, and verified the new bundle before publishing it. All model `local_path` values now point inside the primary project. Approximately 19 GiB of logical model files are present. APFS clones are independent files and share unchanged disk blocks; none is a symlink. No model download or conversion was performed.

The original bundle and worktree were retained as historical evidence. A byte-preserving copy of the 41 MB v2 evaluation run was placed under root `benchmarks/runs.local/transcript-only-v2-evaluation-20261010/` so the root report's local evidence links resolve. Historical metadata still records the location used during that run; it is not runtime configuration.

## Checks

- Root `./run.sh --check`: passed, including pinned packages, model file hashes, index provenance, host requirements, and offline configuration.
- MLX Metal allocation: passed outside the execution sandbox. The sandboxed attempt reported `No Metal device available`; the permitted host execution returned `array([1], dtype=int32)`.
- Root deterministic suite: 1,069 passed, two existing warnings (Starlette/httpx deprecation and expected `BlankAsrSegmentWarning`).
- Dependency and application-import audits: passed. No prohibited framework was added.
- Shell syntax for launcher/setup and `git diff --check`: passed.
- Live service launched with root `./run.sh`; `GET /` returned HTTP 200. Real jobs used only project-root media, environment, code, models, index, and job storage.
- Wheel installation smoke: passed; installed imports, all 703 taxonomy rows, standards manifest, and static UI assets were present.
- Wheel/source-archive content inspection: passed. Model weights, media, `.worktrees`, environments, caches, uploads, and job storage were absent. The source archive includes root launch/setup/import scripts, lockfile, overview, and third-party notices.

The root test count differs from the earlier worktree's 1,086 because the worktree also contained untracked historical trial tests that are not part of the supported runtime distribution. The migrated ASR tests and four new bundle-import boundary tests are included in the root suite.

## Real root-launch outcomes

These are installation smoke checks, not a new accuracy benchmark. Each input is a new extracted excerpt from existing project-folder media; the entire selected excerpt was processed.

| Input | Source range | Result | Export |
| --- | --- | --- | --- |
| Hario Switch — The Best Home Coffee Brewer | 00:30–01:15, 45 seconds | English; `INSUFFICIENT_TOPIC_EVIDENCE`; no final primary or secondaries. Supported primary-role citations covered only 11.12 seconds. | HTTP 409 |
| 5 Things to Know Before Buying a Cartier Tank | 01:00–02:00, 60 seconds | English; primary **Men’s Jewelry and Watches (581)**; no secondaries. Supported primary-role citations covered 48.22 seconds. | HTTP 200; bytes identical to saved `result.json` |

Coffee job: `ca0d6da5-86c5-437e-a5c8-6008e04e83fe`. Watch job: `1d0eeb0b-b133-4b05-8ec8-d3f706104f72`. The watch export SHA-256 is `91e2f065c5ecba786efa56c39830c5c5acd9ac4d7306219a545fd25e8fca0197`. Jobs and excerpt audio remain ignored local artifacts under `data/`.

The temporary verification service is stopped after these checks. Launch with `./run.sh` for interactive use. No additional manual browser transport-reconnect check was performed; the previously accepted substitute and disclosure remain in force.

## Documentation and distribution

[Application overview](../application-overview.md) records current design, requirements, features, models, privacy/storage behavior, limitations, and distribution boundaries. [README](../../README.md) provides project-root setup/launch commands. [Third-party notices](../../THIRD_PARTY_NOTICES.md) records upstream assets and license metadata.

The v2 report's earlier statement that transcript promotion used 70% was corrected with a dated note. Actual transcript code requires more than 50% of usable transcript duration and at least 25% of selected-audio duration. The separate 70% acoustic foreground/genre design remains unenabled. This documentation correction changes no policy code or frozen result.

`.gitignore` now excludes project media, converted/reference assets, local model data, jobs/uploads, smoke evidence, environments, and caches. Source archives use an explicit allowlist; the application wheel continues to exclude model weights. GitHub users acquire the pinned model bundle separately during setup or import a verified existing bundle.

Preparation is local. No remote is configured or repository published. The maintainer still needs to choose the repository/visibility and the application's source license for publication. The existing experimental limitations remain disclosed; no blind accuracy or production-readiness claim is added.

# Terminal Vocal-Uncertainty Completion Probe

**Date:** September 26, 2026 (America/New_York)
**Purpose:** Validate the approved v2 completion amendment on the two full-length
English vocal tracks that previously failed only at their final windows.
**Scope:** Completion behavior and returned model labels; not a category-accuracy
acceptance test.

## Results

| Clip | Status | Final excluded range | Primary | Optional secondary | Runtime |
|---|---|---:|---|---|---:|
| Adele — *Someone Like You* | Complete | 270.00–284.978503 s (`VOCAL_PRESENCE_UNCERTAIN`) | Dating (188) | Marriage and Civil Unions (191); Single Life (200) | 126.61 s |
| Whitney Houston — *I Will Always Love You* | Complete | 270.00–274.041905 s (`VOCAL_PRESENCE_UNCERTAIN`) | Adult Contemporary Music (339) | Prom (173) | 114.43 s |

Both runs detected English, reached 100% progress, and returned no job error.
The uncertain tail remained a transcription-window diagnostic and was not
transcribed or assigned category evidence. Each preceding window had a
successful English transcript; normal classification coverage completed.

### Resource readings

| Clip | Peak process RSS | Peak MLX memory |
|---|---:|---:|
| Adele | 17.73 GB | 19.74 GB |
| Whitney Houston | 15.29 GB | 19.69 GB |

Values are decimal GB (`bytes / 1,000,000,000`). These are probe-process
readings, not a full-application resource acceptance measurement.

## Label assessment and limitations

- Adele's relationship-related labels are plausible for the lyrics, but their
  presence does not establish general classification accuracy.
- Adult Contemporary Music is a strong fit for Whitney Houston's song. Prom is
  less directly supported by the lyrics and should be treated as a questionable
  secondary prediction, not a confirmed human label.
- A nonfatal Whisper tokenizer warning appeared in both runs. Adele also logged
  a skipped blank ASR segment. Neither prevented completion.
- These two observations verify the narrow terminal-window exception only.
  They do not validate arbitrary partial runs, category accuracy, or release
  acceptance. Unit and integration tests continue to cover the fail-closed
  cases, including uncertain-only input, internal uncertainty, multiple
  uncertain windows and other failures.

## Evidence

Raw local probe reports and window journals are retained under the ignored
`benchmarks/runs.local/task13-terminal-probe/` directory:

- Adele job `bc2f8132-b52f-4acb-9552-29704d330e63`
- Whitney Houston job `899d31d5-9062-4980-9c73-08071a30a9ad`

The regression and integration suite passed after the change:
`.venv/bin/python -m pytest -q` — **539 passed**; `git diff --check` — passed.

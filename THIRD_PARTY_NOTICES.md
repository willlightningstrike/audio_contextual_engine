# Third-party assets and distribution

Checked 2026-10-11 using pinned local model metadata and the linked upstream sources. Model weights are local setup assets, excluded from Git and Python distribution artifacts.

| Asset | Upstream source and license metadata |
| --- | --- |
| Qwen3.6-35B-A3B-4bit | [MLX model card](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit), Apache-2.0; [base-model license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE) |
| Qwen3-Embedding-4B-4bit-DWQ | [MLX model card](https://huggingface.co/mlx-community/Qwen3-Embedding-4B-4bit-DWQ), Apache-2.0 |
| Whisper MLX conversion | [Converted card](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16); [OpenAI Whisper MIT license](https://github.com/openai/whisper/blob/main/LICENSE). The pinned conversion card has no separate license field; review conversion redistribution terms before shipping weights. |
| YAMNet | [Pinned source](https://github.com/tensorflow/models/tree/c14bf9ad91962cf189f9f58db2132c06247fcd53/research/audioset/yamnet); [Apache-2.0 license](https://github.com/tensorflow/models/blob/c14bf9ad91962cf189f9f58db2132c06247fcd53/LICENSE), included with the verified converted assets. |
| IAB Content Taxonomy 3.1 | [Pinned TSV](https://github.com/InteractiveAdvertisingBureau/Taxonomies/blob/6dc67c743572e856c7fa86d310cfa844c9fcda85/Content%20Taxonomies/Content%20Taxonomy%203.1.tsv) at commit `6dc67c74`; [Taxonomies repository](https://github.com/InteractiveAdvertisingBureau/Taxonomies), Creative Commons Attribution 3.0 |

Python dependencies retain their upstream licenses in installed distributions; `pyproject.toml` and `uv.lock` identify versions. This notice does not replace upstream license texts or attribution requirements.

The application's own source code is licensed under the MIT License (see `LICENSE`). That license does not cover the third-party models or data above. Personal evaluation media, journals, transcripts, and local exports are excluded from distribution.

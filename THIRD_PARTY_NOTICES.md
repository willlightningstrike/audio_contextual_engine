# Third-party assets and distribution

Checked 2026-10-11 using pinned local model metadata and the linked upstream sources. Model weights are local setup assets, excluded from Git and Python distribution artifacts.

| Asset | Upstream source and license metadata |
| --- | --- |
| Qwen2.5-32B-Instruct-4bit | [MLX model card](https://huggingface.co/mlx-community/Qwen2.5-32B-Instruct-4bit), Apache-2.0; [base-model license](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/LICENSE) |
| Qwen3-Embedding-0.6B-4bit-DWQ | [MLX model card](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ), Apache-2.0 |
| Whisper MLX conversion | [Converted card](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16); [OpenAI Whisper MIT license](https://github.com/openai/whisper/blob/main/LICENSE). The pinned conversion card has no separate license field; review conversion redistribution terms before shipping weights. |
| YAMNet | [Pinned source](https://github.com/tensorflow/models/tree/c14bf9ad91962cf189f9f58db2132c06247fcd53/research/audioset/yamnet); [Apache-2.0 license](https://github.com/tensorflow/models/blob/c14bf9ad91962cf189f9f58db2132c06247fcd53/LICENSE), included with the verified converted assets. |
| IAB Content Taxonomy 3.0 | [Pinned TSV](https://github.com/InteractiveAdvertisingBureau/Taxonomies/blob/0c236b7d73811983bc0e7a314ebb2769acde2291/Content%20Taxonomies/Content%20Taxonomy%203.0.tsv); [implementation guide](https://iabtechlab.com/wp-content/uploads/2021/09/Implementation-Guide-Content-Taxonomy-3-0-pc-Sept2021.pdf), Creative Commons Attribution 3.0 as stated in the guide. Credit: IAB Tech Lab. Source/checksum provenance is in `data/standards/manifest.json`. |

Python dependencies retain their upstream licenses in installed distributions; `pyproject.toml` and `uv.lock` identify versions. This notice does not replace upstream license texts or attribution requirements.

The application's own source code is licensed under the MIT License (see `LICENSE`). That license does not cover the third-party models or data above. Personal evaluation media, journals, transcripts, and local exports are excluded from distribution.

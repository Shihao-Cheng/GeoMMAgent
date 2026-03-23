<p align="center">
  <h1 align="center">🛰️ GeoMMBench & GeoMMAgent: A Multimodal Benchmark and Multi-Agent Framework for GeoScience and Remote Sensing</h1>
</p>

<p align="center">
  <a href="https://github.com/Shihao-Cheng/GeoMMAgent">💻 Code</a> |
  <a href="https://arxiv.org/abs/XXXX.XXXXX">📄 Paper</a> |
  <a href="https://huggingface.co/datasets/AR-X/GeoMMBench">🤗 Dataset</a> |
  <a href="https://geo-mm-agi.github.io">🌐 Project Page</a> |
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CVPR-2026-blue" alt="CVPR 2026">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10+-yellow" alt="Python">
</p>

---

## News

- `2026/03/23`: 🎉 GeoMMBench & GeoMMAgent is accepted by **CVPR 2026**!
- `2026/03/23`: We release the GeoMMBench dataset on [🤗 Hugging Face](https://huggingface.co/datasets/AR-X/GeoMMBench), containing 1,053 expert-level multiple-choice questions.
- `2026/03/23`: Code for GeoMMAgent (coordinator, exec_agents, toolkit) is publicly available.

---

## Introduction

**GeoMMBench** is a comprehensive multimodal question-answering benchmark for geoscience and remote sensing (RS), featuring **1,053 expert-level, image-based multiple-choice questions** covering:

- 🌍 **4 disciplines**: Remote Sensing, Photogrammetry, GIS, GNSS
- 📡 **6 sensor modalities**: Optical, SAR, Hyperspectral, LiDAR, DEM, Thermal
- 🔬 **Diverse task spectrums**: Scene classification, object detection, change detection, spectral analysis, spatial reasoning, and more

![GeoMMBench Overview](assets/benchmark_examples.png)

**GeoMMAgent** is a multi-agent framework following a **plan–execute–evaluate** paradigm, integrating toolkits and agent roles below.

| Toolkit | Capability (this repo) |
|---------|---------------------------|
| 🔧General | Format conversion, filtering, scaling, neural super-resolution (Real-ESRGAN optional), etc. (`toolkit/general.py`) |
| 🔍Knowledge | Web search (short-circuit: DuckDuckGo → Google → Bing → Wikipedia text); optional image search + downloaded evidence images; **GME** text–image similarity to rank/filter candidates (`toolkit/gme_filter.py`, weights in config) |
| 👁️Perception | Scene classification & object detection (**YOLO11**); semantic segmentation (**DeepLabV3+ with Xception**, bundled under `toolkit/deeplabv3plus_xception/`, weights in config) |
| 🧠Reasoning | Reasoning & matching agents (multimodal LLM); option alignment in-agent |

| Agent | Role |
|-------|------|
| Coordinator | Task planning, decomposition, and orchestration (`coordinator/`) |
| Perception (Cls / Det / Seg) | Classification, detection, DeepLab segmentation |
| Search | Retrieval + evidence images for downstream VLMs |
| Reasoning / Matching | Multi-step inference and MCQ alignment |
| Self-Evaluation | Optional quality check (`exec_agents/evaluation/`) |

![GeoMMAgent Framework](assets/agent_framework.png)

---

## Benchmark Results

GeoMMBench evaluates **36+ vision-language models** under zero-shot conditions. GeoMMAgent achieves strong performance.

![Benchmark Results](assets/benchmark_results.png)

See the [paper](https://arxiv.org/abs/XXXX.XXXXX) for full results and analysis.

---

## GeoMMAgent Architecture

```
coordinator/
  ├── coordinator.py        ← Dispatch → sequential execution, multi-image context
  └── prompts.py

exec_agents/
  ├── general/              ← Preprocess agents
  ├── perception/           ← ClsAgent, DetAgent, SegAgent (DeepLab)
  ├── knowledge/            ← SearchAgent (+ evidence images)
  ├── reasoning/            ← ReasoningAgent, MatchingAgent
  └── evaluation/           ← SelfEvaluationAgent (optional)

configs/
  └── GeoMMBench.yaml       ← Coordinator & agents; gme: & SegAgent.deeplab_weights

toolkit/
  ├── general.py
  ├── classification_toolkit.py   ← YOLO11 classification
  ├── detection_toolkit.py        ← YOLO11 OBB detection
  ├── segmentation_toolkit.py   ← DeepLabV3+ Xception inference
  ├── deeplabv3plus_xception/   ← Minimal DeepLab code (from DeepLabV3Plus-Pytorch)
  ├── gme_filter.py             ← GME-style embedding filter (see env / yaml)
  ├── knowledge.py              ← Retrieval hook (optional)
  ├── reasoning.py              ← Placeholder exports (logic in agents)
  ├── super_resolution.py
  └── data_loader.py            ← GeoMMBench parquet loader
```

---

## Quick Start

### Installation

Official repository: **[github.com/Shihao-Cheng/GeoMMAgent](https://github.com/Shihao-Cheng/GeoMMAgent)**.

```bash
git clone https://github.com/Shihao-Cheng/GeoMMAgent.git
cd GeoMMAgent
pip install -r requirements.txt
```

For GPU, install a matching **PyTorch** build from [pytorch.org](https://pytorch.org/) before or after the step above. Segmentation and GME may need `transformers` with versions compatible with your stack (see paper / model cards).

### Environment & configuration

```bash
cp .env_template .env
```

Edit `.env` for API keys. Typical entries:

- **`QWEN_API_KEY`**: Multimodal / text models ([Model Studio](https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key)).
- **Search (optional)**: `GOOGLE_API_KEY` + `SEARCH_ENGINE_ID` for Google Programmable Search; otherwise other engines in the pipeline are used.
- **GME / DeepLab (optional overrides)**: defaults are set from **`configs/GeoMMBench.yaml`** (`gme:` and `SegAgent.deeplab_weights`). Use `.env` only when you need to override.

### Run

From the **GeoMMAgent project root** (paths in `configs/GeoMMBench.yaml` are relative to this root unless absolute):

```bash
pip install -r requirements.txt
cp .env_template .env   # then set API keys

# Single query (plan → execute → trace)
python run/run_geomm.py --single "这张遥感图像中有几架飞机？" --image /path/to/image.png

# Benchmark on GeoMMBench parquet (place or symlink file at datasets/validation.parquet, or pass a path)
python run/run_geomm.py --bench datasets/validation.parquet

python run/run_geomm.py --bench datasets/validation.parquet --limit 5
```

Parallel benchmark: `python run/run_benchmark_parallel.py --parquet datasets/validation.parquet` (see `run/run_benchmark_parallel.py` docstring). Optional: `bash run_benchmark.sh` with `PARQUET` / `WORKERS` env vars.

**Layout:** put model weights under `weights/`; for neural super-resolution, clone the Real-ESRGAN inference repo into `Real-ESRGAN-master/` at the repo root (see `SuperResolutionAgent` in `configs/GeoMMBench.yaml`). YOLO debug outputs go to `yolo_out/` (gitignored).

### Model configuration

Default coordinator and per-agent models are defined in **`configs/GeoMMBench.yaml`** (`coordinator.model`, `agents.*.model`). Entry point **`run/run_geomm.py`** loads this file via `load_config()` and `create_model_from_config()`.

---

## Dataset

GeoMMBench is on [🤗 Hugging Face](https://huggingface.co/datasets/AR-X/GeoMMBench):

```python
from datasets import load_dataset
ds = load_dataset("GeoMM/GeoMMBench")
```

Each sample contains: `image`, `question`, `options (A/B/C/D)`, `answer`.

For `run/run_geomm.py --bench` and `run/run_benchmark_parallel.py`, export or symlink a Parquet file to **`datasets/validation.parquet`** (default in `configs/GeoMMBench.yaml`) or pass `--parquet /your/path.parquet`.

---

## Model weights

| Component | Notes | Hugging Face |
|-----------|--------|--------------|
| YOLO11-cls | Scene classification | [GeoMM/yolo11-cls-sft](https://huggingface.co/GeoMM/yolo11-cls-millionaid) |
| YOLO11-obb | DOTA-style detection | [GeoMM/yolo11-obb-sft](https://huggingface.co/GeoMM/yolo11-obb-dotav2) |
| DeepLabV3+ (LoveDA) | Semantic segmentation (Xception backbone, **same family** as `toolkit/deeplabv3plus_xception`) | [GeoMM/deeplabv3plus-loveda](https://huggingface.co/GeoMM/deeplabv3plus-loveda) |
| GME | Multimodal embedding filter for search candidates | Local path in **`configs/GeoMMBench.yaml`** → `gme.model_path` (e.g. `weights/gme-Qwen2-VL-2B-Instruct`) |

Place downloaded files under **`weights/`** (or use absolute paths) and align **`SegAgent.deeplab_weights`** (`path`, `num_classes`, `output_stride`) with how each checkpoint was trained. Classification/detection paths are set in **`configs/GeoMMBench.yaml`** under each agent.

---

## Extending GeoMMAgent

Training-free extension:

1. Add tools under `toolkit/` and bind them in `exec_agents/` (`BaseExecAgent`, `get_tools()`).
2. Register agents in **`configs/GeoMMBench.yaml`** (`agents.<Name>.enabled`) and ensure `configs/loader.py` `AGENT_REGISTRY` includes your class.
3. Run via `run/run_geomm.py` which calls `build_agents_from_config()`.

---

## License

The GeoMMBench dataset is distributed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
The code is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## Citation

```bibtex
@inproceedings{xiao2026geomm,
  title={GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing},
  author={Xiao, Aoran and Cheng, Shihao and Xu, Yonghao and Ren, Yexian and Chen, Hongruixuan and Yokoya, Naoto},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

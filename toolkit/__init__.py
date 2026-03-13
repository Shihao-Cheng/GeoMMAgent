# toolkit/__init__.py
# 工具总览：区分"框架现成"与"自定义 RS 专用"
#
# ══════════════════════════════════════════════════════════════
# 一、CAMEL 框架已内置（直接 import 使用，无需重复定义）
# ══════════════════════════════════════════════════════════════
#
# ── 搜索 ──────────────────────────────────────────────────────
#   from camel.toolkits import SearchToolkit
#   tools = SearchToolkit().get_tools()
#   包含: search_duckduckgo / search_google / search_wiki / search_bing
#
# ── 图像分析 ──────────────────────────────────────────────────
#   from camel.toolkits import ImageAnalysisToolkit
#   tools = ImageAnalysisToolkit().get_tools()
#   包含: 用 VLM 对图片做通用图文问答、描述、OCR 等
#
# ── 视频分析 ──────────────────────────────────────────────────
#   from camel.toolkits import VideoAnalysisToolkit
#
# ── RAG 检索（文档/网页 → 向量库 → 语义检索）────────────────
#   from camel.retrievers import AutoRetriever       # 一行搞定全流程
#   from camel.retrievers import VectorRetriever     # 自定义 Embedding + Storage
#   from camel.embeddings import OpenAIEmbedding, SentenceTransformerEncoder
#   from camel.storages import QdrantStorage, ChromaStorage
#
# ── 网页 / 文档加载器 ─────────────────────────────────────────
#   from camel.loaders import JinaURLReader          # URL → Markdown
#   from camel.loaders import UnstructuredIO         # PDF/HTML 解析+分块
#   from camel.loaders import MistralReader          # PDF OCR
#
# ── 代码 / 终端执行 ───────────────────────────────────────────
#   from camel.toolkits import CodeExecutionToolkit, TerminalToolkit
#
# ── 学术检索 ──────────────────────────────────────────────────
#   from camel.toolkits import ArxivToolkit, GoogleScholarToolkit
#
# ══════════════════════════════════════════════════════════════
# 二、自定义 RS 专用工具（在下方各文件中实现）
# ══════════════════════════════════════════════════════════════
#
#   perception.py  → classify_scene / detect_objects / segment_image
#                    (YOLO11 分类/检测, DeepLabv3+ 分割，框架无内置)
#
#   knowledge.py   → retrieve_multimodal
#                    (GME 多模态检索，框架无内置；RAG 检索直接用 AutoRetriever)
#
#   reasoning.py   → match_answer_to_choices / analyze_spatial_change
#                    (RS 多选题语义匹配 + 时序变化检测，框架无内置)
#
#   general.py     → convert_format / patch_tile / patch_merge / filter_image
#                    / crop_image / scale_image / super_resolve / count_area / count_boxes
#                    (通用预处理/后处理工具集)
#
#   super_resolution.py → run_super_resolution（占位，待接入专有 SR 模型）

from .perception import get_perception_tools
from .knowledge import get_knowledge_tools
from .reasoning import get_reasoning_tools
from .general import get_general_tools
from .data_loader import load_benchmark, BenchmarkSample

__all__ = [
    "get_perception_tools",
    "get_knowledge_tools",
    "get_reasoning_tools",
    "get_general_tools",
    "load_benchmark",
    "BenchmarkSample",
]

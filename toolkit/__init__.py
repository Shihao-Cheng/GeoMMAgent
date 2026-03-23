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
#   classification_toolkit.py / detection_toolkit.py / segmentation_toolkit.py
#   → YOLO cls / YOLO det / DeepLab seg（ClsAgent, DetAgent, SegAgent）
#   deeplabv3plus_xception/ → DeepLabV3+ Xception 网络（自 DeepLabV3Plus-Pytorch 精简拷贝）
#
#   knowledge.py   → retrieve_multimodal
#                    (GME 多模态检索，框架无内置；RAG 检索直接用 AutoRetriever)
#   gme_filter.py  → filter_evidence_candidates（检索候选过滤；占位，可接 CLIP/GME）
#
#   reasoning.py   → get_reasoning_tools()（当前为空；匹配与推理在 Agent 内完成）
#
#   general.py     → get_format_conversion_tools / get_image_filter_tools /
#                    get_scale_tools / get_super_resolution_tools
#
#   super_resolution.py → run_super_resolution（双线性，无权重时回退）
#   neural_sr.py       → 神经超分（YAML：sr_weights / sr_repo）

from .knowledge import get_knowledge_tools
# MCP: ``python -m geomm_mcp.cli --toolkit classification`` (CAMEL ``@MCPServer``).
from .reasoning import get_reasoning_tools
from .general import (
    get_format_conversion_tools,
    get_image_filter_tools,
    get_scale_tools,
    get_super_resolution_tools,
    get_general_tools,
)
from .data_loader import (
    load_benchmark,
    BenchmarkSample,
    get_benchmark_sample_by_id,
    get_benchmark_sample_from_jsonl,
)
from .evaluation_metrics import (
    run_self_evaluation_metrics,
    evaluate_trace_with_metrics,
    format_metrics_checklist,
    SELF_EVAL_DIMENSIONS,
)

__all__ = [
    "get_knowledge_tools",
    "get_reasoning_tools",
    "get_format_conversion_tools",
    "get_image_filter_tools",
    "get_scale_tools",
    "get_super_resolution_tools",
    "get_general_tools",
    "load_benchmark",
    "BenchmarkSample",
    "get_benchmark_sample_by_id",
    "get_benchmark_sample_from_jsonl",
    "run_self_evaluation_metrics",
    "evaluate_trace_with_metrics",
    "format_metrics_checklist",
    "SELF_EVAL_DIMENSIONS",
]

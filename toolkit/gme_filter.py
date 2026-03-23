# toolkit/gme_filter.py
# GME: embed query text vs candidate image URLs, rank by similarity, then truncate.

from __future__ import annotations

import logging
import os
import pathlib
import threading
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_gme_model = None
_gme_lock = threading.Lock()

# 与 Alibaba-NLP/gme-Qwen2-VL-2B-Instruct 示例一致
_DEFAULT_INSTRUCTION = (
    "Find an image that matches the given text."
)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _default_model_path() -> str:
    return os.environ.get(
        "GEOMM_GME_MODEL_PATH",
        str(_repo_root() / "weights" / "gme-Qwen2-VL-2B-Instruct"),
    )


def _get_gme_model():
    """进程内单例；首次调用时加载。"""
    global _gme_model
    with _gme_lock:
        if _gme_model is not None:
            return _gme_model
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:
            raise RuntimeError(f"GME requires torch and transformers: {e}") from e

        path = _default_model_path()
        if not os.path.isdir(path):
            raise FileNotFoundError(f"GME model path not found: {path}")

        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        device_map = "cuda" if use_cuda else "cpu"

        logger.info("Loading GME from %s (device_map=%s)", path, device_map)
        _gme_model = AutoModel.from_pretrained(
            path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        return _gme_model


def _embed_images_batched(model, urls: List[str], batch_size: int) -> Any:
    """分批图像嵌入，避免一次性 OOM。"""
    import torch

    parts = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i : i + batch_size]
        emb = model.get_image_embeddings(images=batch, is_query=False)
        parts.append(emb)
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=0)


def _scores_text_vs_images(
    question: str,
    urls: List[str],
    instruction: str,
    batch_size: int,
) -> Any:
    """返回 shape (N,) 的相似度分数（越高越相关）。"""
    import torch

    model = _get_gme_model()
    e_q = model.get_text_embeddings(texts=[question], instruction=instruction)
    e_img = _embed_images_batched(model, urls, batch_size=batch_size)

    if hasattr(e_q, "float"):
        e_q = e_q.float()
    if hasattr(e_img, "float"):
        e_img = e_img.float()

    sim = e_q @ e_img.T
    if hasattr(sim, "squeeze"):
        sim = sim.squeeze(0)
    if hasattr(sim, "cpu"):
        sim = sim.cpu()
    return sim.detach().flatten()


def filter_evidence_candidates(
    question: str,
    task_image_path: Optional[str],
    candidates: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    用 GME：题文 + instruction → 文本向量；候选图 URL → 图像向量；点积排序后取 top_k。

    task_image_path 预留；当前稳妥方案仅用文本 query（与官方 t2i 示例一致）。

    环境变量：
        GEOMM_GME_DISABLE=1 — 不加载 GME，仅按原序截断 top_k
        GEOMM_GME_NOOP=1 — 不做截断，原样返回
        GEOMM_GME_MODEL_PATH — 权重目录
        GEOMM_GME_TOP_K — 默认保留条数（filter 未传 top_k 时）
        GEOMM_GME_INSTRUCTION — 覆盖默认 instruction
        GEOMM_GME_IMG_BATCH — 图像嵌入 batch 大小（默认 8）
    """
    if not candidates:
        return []

    if os.environ.get("GEOMM_GME_NOOP", "").strip().lower() in ("1", "true", "yes"):
        return list(candidates)

    k = top_k
    if k is None:
        raw = os.environ.get("GEOMM_GME_TOP_K", "").strip()
        k = int(raw) if raw.isdigit() else 12
    k = max(1, k)

    if os.environ.get("GEOMM_GME_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        out = candidates[:k]
        logger.debug("gme_filter (disabled): %d -> %d", len(candidates), len(out))
        return out

    indexed: List[tuple] = []
    urls: List[str] = []
    for c in candidates:
        u = c.get("url")
        if not u or not isinstance(u, str):
            continue
        indexed.append((i, c))
        urls.append(u)

    if not urls:
        return candidates[:k]

    instruction = os.environ.get("GEOMM_GME_INSTRUCTION", "").strip() or _DEFAULT_INSTRUCTION
    batch = int(os.environ.get("GEOMM_GME_IMG_BATCH", "8") or "8")
    batch = max(1, batch)

    try:
        scores = _scores_text_vs_images(
            question.strip() or ".",
            urls,
            instruction=instruction,
            batch_size=batch,
        )

        arr = scores.numpy() if hasattr(scores, "numpy") else np.asarray(scores)
        order = np.argsort(-arr)
        ranked: List[Dict[str, Any]] = []
        for j in order:
            ji = int(j)
            ranked.append(indexed[ji][1])
        out = ranked[:k]
        logger.debug(
            "gme_filter (GME): scored %d urls -> keep %d",
            len(urls),
            len(out),
        )
        return out
    except Exception as e:
        logger.warning("GME filter failed, fallback to truncate: %s", e)
        return candidates[:k]

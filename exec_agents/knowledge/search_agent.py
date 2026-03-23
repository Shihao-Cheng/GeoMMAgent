# exec_agents/knowledge/search_agent.py
# 搜索：从 subtask 抽 query → 图搜 → GME 占位过滤 → 下载证据图；文本短路检索保留。

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit
from camel.types import RoleType

from exec_agents.base import BaseExecAgent
from .prompts import SEARCH_SYSTEM_PROMPT, SEARCH_WORKER_DESC

logger = logging.getLogger(__name__)

_toolkit = SearchToolkit()
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_EVIDENCE_ROOT = _PROJECT_ROOT / "temp_search_evidence"


def _max_evidence_images() -> int:
    raw = os.environ.get("GEOMM_MAX_EVIDENCE_IMAGES", "").strip()
    return int(raw) if raw.isdigit() else 4


def _format_result_items(
    source_name: str,
    items: Any,
    max_results: int,
) -> Optional[str]:
    if not items:
        return None
    if isinstance(items, dict):
        if items.get("error"):
            return None
        items = items.get("results", [])
    if not isinstance(items, list):
        text = str(items)[:2000]
        return f"[{source_name}]\n{text}"

    rows: List[str] = []
    for item in items[:max_results]:
        if not isinstance(item, dict):
            continue
        if item.get("error"):
            continue
        title = item.get("title", "")
        desc = item.get("description") or item.get("snippet") or item.get(
            "long_description", ""
        )
        link = item.get("link") or item.get("url", "")
        rows.append(f"- {title}: {desc} ({link})")
    if not rows:
        return None
    return f"[Search results from {source_name}]\n" + "\n".join(rows)


def _google_block(q: str, max_results: int) -> Optional[str]:
    fn = getattr(_toolkit, "search_google", None)
    if not fn or not os.getenv("GOOGLE_API_KEY") or not os.getenv("SEARCH_ENGINE_ID"):
        return None
    try:
        raw = fn(q, num_result_pages=max_results)
        return _format_result_items("google (web)", raw, max_results)
    except Exception as e:
        logger.debug("Google search failed: %s", e)
        return None


def _google_images_block(q: str, max_results: int) -> Optional[str]:
    key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("SEARCH_ENGINE_ID")
    if not key or not cx:
        return None
    n = min(max(1, max_results), 10)
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": key,
                "cx": cx,
                "q": q,
                "searchType": "image",
                "num": n,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            logger.debug("Google image API error: %s", data.get("error"))
            return None
        items = data.get("items") or []
        rows: List[str] = []
        for it in items[:max_results]:
            if not isinstance(it, dict):
                continue
            title = it.get("title", "")
            page = it.get("link", "")
            img_meta = it.get("image") or {}
            thumb = img_meta.get("thumbnailLink") or img_meta.get("contextLink") or ""
            if title or thumb or page:
                rows.append(f"- {title}\n  image: {thumb}\n  page: {page}")
        if not rows:
            return None
        return "[Search results from google (images)]\n" + "\n".join(rows)
    except Exception as e:
        logger.debug("Google image search failed: %s", e)
        return None


def _google_raw_image_items(q: str, max_results: int) -> List[Dict[str, Any]]:
    key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("SEARCH_ENGINE_ID")
    if not key or not cx:
        return []
    n = min(max(1, max_results), 10)
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": key,
                "cx": cx,
                "q": q,
                "searchType": "image",
                "num": n,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            return []
        out: List[Dict[str, Any]] = []
        for it in (data.get("items") or [])[:max_results]:
            if not isinstance(it, dict):
                continue
            img_meta = it.get("image") or {}
            url = img_meta.get("thumbnailLink") or img_meta.get("contextLink")
            if not url:
                continue
            out.append({
                "url": url,
                "title": it.get("title", ""),
                "page_url": it.get("link", ""),
            })
        return out
    except Exception as e:
        logger.debug("Google raw image items: %s", e)
        return []


def _wiki_block(q: str) -> Optional[str]:
    fn = getattr(_toolkit, "search_wiki", None)
    if not fn:
        return None
    try:
        entity = q if len(q) <= 400 else q[:300].rsplit(" ", 1)[0]
        summary = fn(entity)
        if not summary or not isinstance(summary, str):
            return None
        low = summary.lower()
        if "no page in wikipedia" in low[:120]:
            return None
        cap = 2500
        text = summary if len(summary) <= cap else summary[:cap] + "…"
        return f"[Wikipedia summary]\n{text}"
    except Exception as e:
        logger.debug("Wikipedia search failed: %s", e)
        return None


def _ddg_raw_image_items(q: str, max_results: int) -> List[Dict[str, Any]]:
    fn = getattr(_toolkit, "search_duckduckgo", None)
    if not fn:
        return []
    try:
        raw = fn(q, source="images", max_results=max_results)
        if isinstance(raw, dict) and raw.get("error"):
            return []
        items = raw if isinstance(raw, list) else raw.get("results", [])
        if not isinstance(items, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in items[:max_results]:
            if not isinstance(item, dict) or item.get("error"):
                continue
            url = item.get("image") or item.get("url")
            if not url:
                continue
            out.append({
                "url": url,
                "title": item.get("title", ""),
                "page_url": item.get("url", ""),
            })
        return out
    except Exception as e:
        logger.debug("DDG raw image items: %s", e)
        return []


def _ddg_block(q: str, max_results: int) -> Optional[str]:
    fn = getattr(_toolkit, "search_duckduckgo", None)
    if not fn:
        return None
    try:
        raw = fn(q, source="text", max_results=max_results)
        return _format_result_items("duckduckgo (web)", raw, max_results)
    except Exception as e:
        logger.debug("DuckDuckGo search failed: %s", e)
        return None


def _ddg_images_block(q: str, max_results: int) -> Optional[str]:
    fn = getattr(_toolkit, "search_duckduckgo", None)
    if not fn:
        return None
    try:
        raw = fn(q, source="images", max_results=max_results)
        if isinstance(raw, dict) and raw.get("error"):
            return None
        items = raw if isinstance(raw, list) else raw.get("results", [])
        if not isinstance(items, list):
            return None
        rows: List[str] = []
        for item in items[:max_results]:
            if not isinstance(item, dict) or item.get("error"):
                continue
            title = item.get("title", "")
            img_url = item.get("image", "")
            page = item.get("url", "")
            src = item.get("source", "")
            rows.append(
                f"- {title}\n  image: {img_url}\n  page: {page}"
                + (f"\n  via: {src}" if src else "")
            )
        if not rows:
            return None
        return "[Search results from duckduckgo (images)]\n" + "\n".join(rows)
    except Exception as e:
        logger.debug("DuckDuckGo image search failed: %s", e)
        return None


def _bing_block(q: str, max_results: int) -> Optional[str]:
    fn = getattr(_toolkit, "search_bing", None)
    if not fn:
        return None
    try:
        raw = fn(q)
        return _format_result_items("bing", raw, max_results)
    except Exception as e:
        logger.debug("Bing search failed: %s", e)
        return None


def force_search(query: str, max_results: int = 5) -> str:
    """文本检索短路：DDG→Google→Bing→Wiki；有即停。"""
    q = (query or "").strip()
    if not q:
        return "[No search results available]"

    steps = [
        lambda: _ddg_block(q, max_results),
        lambda: _ddg_images_block(q, max_results),
        lambda: _google_block(q, max_results),
        lambda: _google_images_block(q, max_results),
        lambda: _bing_block(q, max_results),
        lambda: _wiki_block(q),
    ]
    for fn in steps:
        try:
            text = fn()
        except Exception as e:
            logger.debug("Search step failed: %s", e)
            text = None
        if text:
            return text

    return "[No search results available]"


def get_image_search_metadata_json(query: str, max_results: int = 5) -> str:
    """
    Return JSON listing image-search candidates (``url``, ``title``, ``page_url``)
    from DuckDuckGo and Google Custom Search. Does not download images; callers
    filter, rank, or fetch in a later step (see ``run_search_evidence_pipeline``).
    """
    q = (query or "").strip()
    if not q:
        return json.dumps(
            {"ok": False, "error": "empty query", "items": []},
            ensure_ascii=False,
        )
    cap = max(1, min(int(max_results), 20))
    items: List[Dict[str, Any]] = []
    seen: set = set()
    for fn in (_ddg_raw_image_items, _google_raw_image_items):
        try:
            for it in fn(q, cap):
                if not isinstance(it, dict):
                    continue
                u = (it.get("url") or "").strip()
                if not u or u in seen:
                    continue
                seen.add(u)
                items.append({
                    "url": u,
                    "title": it.get("title", ""),
                    "page_url": it.get("page_url", ""),
                })
                if len(items) >= cap:
                    break
        except Exception as e:
            logger.debug("get_image_search_metadata_json: %s", e)
        if len(items) >= cap:
            break
    return json.dumps(
        {
            "ok": True,
            "query": q,
            "items": items,
            "note": "URLs only; download or GME-filter in a downstream step.",
        },
        ensure_ascii=False,
    )


def pack_search_evidence_payload(text_block: str, image_metadata_json: str) -> str:
    """
    Merge a text search snippet with structured image metadata (JSON string) for
    retrieval or matching agents. Does not download images.
    """
    meta_obj: Any
    if not (image_metadata_json or "").strip():
        meta_obj = {}
    else:
        try:
            meta_obj = json.loads(image_metadata_json)
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "ok": False,
                    "error": "image_metadata_json is not valid JSON",
                    "detail": str(e)[:200],
                },
                ensure_ascii=False,
            )
    if not isinstance(meta_obj, dict):
        meta_obj = {"value": meta_obj}
    return json.dumps(
        {
            "ok": True,
            "text_block": text_block or "",
            "image_metadata": meta_obj,
        },
        ensure_ascii=False,
        indent=2,
    )


def _parse_queries_json(text: str) -> Optional[List[str]]:
    text = text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    try:
        data = json.loads(text)
        qs = data.get("queries")
        if isinstance(qs, list):
            return [str(x).strip() for x in qs if str(x).strip()]
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\"queries\"[\s\S]*\}", text)
        if m:
            try:
                data = json.loads(m.group())
                qs = data.get("queries")
                if isinstance(qs, list):
                    return [str(x).strip() for x in qs if str(x).strip()]
            except json.JSONDecodeError:
                pass
    return None


def extract_search_queries(
    chat_agent: ChatAgent,
    subtask: str,
    original_task: str,
) -> List[str]:
    """从 subtask + 原题中抽取用于图搜的短 query 列表。"""
    prompt = (
        "You extract short web IMAGE search queries to retrieve helpful reference pictures.\n"
        'Return ONLY valid JSON: {"queries": ["query1", "query2"]}\n'
        "Use 1 to 3 queries. They may be English or Chinese.\n\n"
        f"Subtask:\n{subtask}\n\nFull task / question:\n{original_task}\n"
    )
    msg = BaseMessage(
        role_name="user",
        role_type=RoleType.USER,
        meta_dict={},
        content=prompt,
    )
    try:
        resp = chat_agent.step(msg)
        raw = (resp.msgs[0].content if resp.msgs else "").strip()
        parsed = _parse_queries_json(raw)
        if parsed:
            return parsed[:3]
    except Exception as e:
        logger.debug("extract_search_queries failed: %s", e)
    fallback = subtask.strip()[:400] or original_task.strip()[:400]
    return [fallback] if fallback else ["remote sensing"]


def _collect_image_candidates_for_queries(
    queries: List[str],
    per_query: int = 6,
) -> List[Dict[str, Any]]:
    """多 query 拉取图搜 URL，去重。"""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for q in queries:
        label = q
        for src_name, items in (
            ("ddg", _ddg_raw_image_items(q, per_query)),
            ("google", _google_raw_image_items(q, per_query)),
        ):
            for it in items:
                url = it.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                out.append({
                    "label": label,
                    "url": url,
                    "title": it.get("title", ""),
                    "page_url": it.get("page_url", ""),
                    "source": src_name,
                })
    return out


def _download_image_to_dir(url: str, out_dir: pathlib.Path) -> Optional[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "GeoMMAgent/1.0 (research)"},
        )
        r.raise_for_status()
        if len(r.content) > 5 * 1024 * 1024:
            return None
        ext = ".jpg"
        ct = (r.headers.get("content-type") or "").lower()
        if "png" in ct:
            ext = ".png"
        elif "webp" in ct:
            ext = ".webp"
        elif "gif" in ct:
            ext = ".gif"
        name = f"{uuid.uuid4().hex}{ext}"
        path = out_dir / name
        path.write_bytes(r.content)
        return str(path.resolve())
    except Exception as e:
        logger.debug("download failed %s: %s", url[:80], e)
        return None


def run_search_evidence_pipeline(
    chat_agent: ChatAgent,
    subtask: str,
    original_task: str,
    image_path: Optional[str],
    max_evidence: Optional[int] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    抽 query → 图搜候选 → GME 占位过滤 → 下载证据图。

    Returns:
        (text_block_for_user_message, evidence list)
        evidence 每项: ``label``, ``path``, ``url``（path 为本地文件）。

    使用独立 ChatAgent 抽 query，避免污染 SearchAgent 主对话。
    """
    max_ev = max_evidence if max_evidence is not None else _max_evidence_images()
    extractor = ChatAgent(
        system_message=(
            "You extract image search queries for retrieval. "
            "Reply with JSON only."
        ),
        model=chat_agent.model_backend,
    )
    queries = extract_search_queries(extractor, subtask, original_task)

    from toolkit.gme_filter import filter_evidence_candidates

    candidates = _collect_image_candidates_for_queries(queries, per_query=8)
    filtered = filter_evidence_candidates(
        original_task,
        image_path,
        candidates,
        top_k=max(16, max_ev * 4),
    )

    _EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    session_dir = _EVIDENCE_ROOT / uuid.uuid4().hex
    evidence: List[Dict[str, str]] = []

    for c in filtered:
        if len(evidence) >= max_ev:
            break
        p = _download_image_to_dir(c["url"], session_dir)
        if p:
            evidence.append({
                "label": c.get("label", ""),
                "path": p,
                "url": c.get("url", ""),
            })

    lines: List[str] = [
        "Search evidence (structured):",
        f"- Extracted image queries: {queries}",
    ]

    if not evidence:
        fb = force_search(queries[0] if queries else subtask)
        lines.append("No downloadable images after GME filter; text-only search:")
        lines.append(fb)
        return "\n".join(lines), []

    lines.append(
        "Downloaded reference images are attached after the task image in multimodal order; "
        "each label matches the search query used for that image."
    )
    for i, ev in enumerate(evidence, start=1):
        lines.append(
            f"  - evidence index {i}: label (query) = {ev['label']!r} | file = {ev['path']}"
        )

    return "\n".join(lines), evidence


class SearchAgent(BaseExecAgent):
    """Web search worker; the coordinator calls :func:`run_search_evidence_pipeline` to attach evidence."""

    SYSTEM_PROMPT = SEARCH_SYSTEM_PROMPT
    WORKER_DESCRIPTION = SEARCH_WORKER_DESC

    def get_tools(self) -> List[FunctionTool]:
        return []

    @staticmethod
    def force_search(query: str, max_results: int = 5) -> str:
        return force_search(query, max_results=max_results)

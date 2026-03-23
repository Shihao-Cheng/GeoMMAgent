# toolkit/perception_io.py
# 感知工具（分类 / 检测 / 分割）统一的 JSON 返回外壳

from __future__ import annotations

import json
from typing import Any, Dict, Optional

SCHEMA_VERSION = "1.0"

TOOL_CLASSIFICATION = "classification"
TOOL_DETECTION = "detection"
TOOL_SEGMENTATION = "segmentation"


def wrap_ok(
    tool: str,
    data: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    body: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "ok": True,
        "error": None,
        "data": data,
    }
    if meta:
        body["meta"] = meta
    return json.dumps(body, ensure_ascii=False, indent=2)


def wrap_err(
    tool: str,
    message: str,
    code: str = "ERROR",
) -> str:
    body = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "ok": False,
        "error": {"code": code, "message": message},
        "data": None,
    }
    return json.dumps(body, ensure_ascii=False, indent=2)

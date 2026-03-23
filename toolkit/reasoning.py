# toolkit/reasoning.py
# 多选题对齐与复杂推理由 MatchingAgent / ReasoningAgent（LLM）完成，不通过占位工具函数暴露。
# 保留 get_reasoning_tools() 供 toolkit 包导出，便于后续接入真实推理工具时再扩展。

from __future__ import annotations

from typing import List

from camel.toolkits import FunctionTool


def get_reasoning_tools() -> List[FunctionTool]:
    """当前无独立推理 FunctionTool；选项匹配与推断在 Agent 对话内完成。"""
    return []

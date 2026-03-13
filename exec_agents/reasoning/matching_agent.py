# exec_agents/reasoning/matching_agent.py
# 选项匹配 Agent — 自由文本答案 → A/B/C/D 对齐

from typing import List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import MATCHING_SYSTEM_PROMPT, MATCHING_WORKER_DESC


class MatchingAgent(BaseExecAgent):
    """
    选项匹配 Agent。

    将其他 Agent 输出的自由文本推理结果
    与多选题选项做语义对齐，输出最终选项字母。
    """

    SYSTEM_PROMPT = MATCHING_SYSTEM_PROMPT
    WORKER_DESCRIPTION = MATCHING_WORKER_DESC

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.reasoning import get_reasoning_tools
            for tool in get_reasoning_tools():
                if "match" in tool.get_function_name():
                    tools.append(tool)
        except ImportError:
            pass
        return tools

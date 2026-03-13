# exec_agents/general/general_agent.py
# 通用预处理/后处理 Agent

from typing import List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import GENERAL_SYSTEM_PROMPT, GENERAL_WORKER_DESC


class GeneralAgent(BaseExecAgent):
    """
    通用工具 Agent，负责遥感影像的预处理与后处理。

    覆盖能力：格式转换、切片/合并、滤波、裁剪、缩放、
    超分辨率、面积统计、框计数。
    """

    SYSTEM_PROMPT = GENERAL_SYSTEM_PROMPT
    WORKER_DESCRIPTION = GENERAL_WORKER_DESC

    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
        system_prompt_override: Optional[str] = None,
    ):
        super().__init__(model=model, system_prompt_override=system_prompt_override)

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.general import get_general_tools
            tools.extend(get_general_tools())
        except Exception:
            pass
        return tools

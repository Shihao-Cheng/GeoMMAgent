# exec_agents/reasoning/reasoning_agent.py
# 多步推理 Agent — 空间时序分析、证据整合、最终推断
#
from typing import List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import REASONING_SYSTEM_PROMPT, REASONING_WORKER_DESC


class ReasoningAgent(BaseExecAgent):
    """
    多步推理 Agent。

    整合感知输出和检索知识，进行逻辑推断，
    支持空间时序分析等复杂推理任务。
    """

    SYSTEM_PROMPT = REASONING_SYSTEM_PROMPT
    WORKER_DESCRIPTION = REASONING_WORKER_DESC

    def get_tools(self) -> List[FunctionTool]:
        return []

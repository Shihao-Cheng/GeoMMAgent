# exec_agents/general/scale_agent.py

from typing import List

from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import SCALE_SYSTEM_PROMPT, SCALE_WORKER_DESC


class ScaleAgent(BaseExecAgent):
    """缩放；scale_factor 由 VLM 根据任务决定。"""

    SYSTEM_PROMPT = SCALE_SYSTEM_PROMPT
    WORKER_DESCRIPTION = SCALE_WORKER_DESC

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.general import get_scale_tools
            tools.extend(get_scale_tools())
        except Exception:
            pass
        return tools

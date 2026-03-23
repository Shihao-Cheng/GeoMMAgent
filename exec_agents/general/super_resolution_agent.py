# exec_agents/general/super_resolution_agent.py

from typing import List

from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import SUPER_RESOLUTION_SYSTEM_PROMPT, SUPER_RESOLUTION_WORKER_DESC


class SuperResolutionAgent(BaseExecAgent):
    """超分辨率；倍率由 VLM 根据任务决定。"""

    SYSTEM_PROMPT = SUPER_RESOLUTION_SYSTEM_PROMPT
    WORKER_DESCRIPTION = SUPER_RESOLUTION_WORKER_DESC

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.general import get_super_resolution_tools
            tools.extend(get_super_resolution_tools())
        except Exception:
            pass
        return tools

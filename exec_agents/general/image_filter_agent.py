# exec_agents/general/image_filter_agent.py

from typing import List

from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import IMAGE_FILTER_SYSTEM_PROMPT, IMAGE_FILTER_WORKER_DESC


class ImageFilterAgent(BaseExecAgent):
    """仅滤波；method 由 VLM 根据子任务决定。"""

    SYSTEM_PROMPT = IMAGE_FILTER_SYSTEM_PROMPT
    WORKER_DESCRIPTION = IMAGE_FILTER_WORKER_DESC

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.general import get_image_filter_tools
            tools.extend(get_image_filter_tools())
        except Exception:
            pass
        return tools

# exec_agents/general/format_conversion_agent.py

from typing import List

from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import FORMAT_CONVERSION_SYSTEM_PROMPT, FORMAT_CONVERSION_WORKER_DESC


class FormatConversionAgent(BaseExecAgent):
    """仅格式转换；target_format 由 VLM 根据子任务决定。"""

    SYSTEM_PROMPT = FORMAT_CONVERSION_SYSTEM_PROMPT
    WORKER_DESCRIPTION = FORMAT_CONVERSION_WORKER_DESC

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.general import get_format_conversion_tools
            tools.extend(get_format_conversion_tools())
        except Exception:
            pass
        return tools

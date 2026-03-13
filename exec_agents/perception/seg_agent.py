# exec_agents/perception/seg_agent.py
# 语义分割 Agent — DeepLabv3+ + VLM 推理

from typing import List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import SEG_SYSTEM_PROMPT, SEG_WORKER_DESC


class SegAgent(BaseExecAgent):
    """
    遥感语义分割 Agent。

    使用 DeepLabv3+ (LoveDA) 做像素级分类，
    返回各类别面积占比，再结合 VLM 推理输出最终选项。
    """

    SYSTEM_PROMPT = SEG_SYSTEM_PROMPT
    WORKER_DESCRIPTION = SEG_WORKER_DESC

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.perception import get_perception_tools
            for tool in get_perception_tools():
                if "segment" in tool.get_function_name():
                    tools.append(tool)
        except ImportError:
            pass
        return tools

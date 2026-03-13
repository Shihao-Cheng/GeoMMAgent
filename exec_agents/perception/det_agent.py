# exec_agents/perception/det_agent.py
# 目标检测 Agent — YOLO11-OBB + VLM 推理

from typing import List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import DET_SYSTEM_PROMPT, DET_WORKER_DESC


class DetAgent(BaseExecAgent):
    """
    遥感目标检测 Agent。

    使用 YOLO11-OBB (DOTA-v2, 18 类) 做有向目标检测，
    返回类别计数，再结合 VLM 推理输出最终选项。
    """

    SYSTEM_PROMPT = DET_SYSTEM_PROMPT
    WORKER_DESCRIPTION = DET_WORKER_DESC

    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
        system_prompt_override: Optional[str] = None,
        yolo_model_path: str = "weights/yolo11s-obb.pt",
    ):
        super().__init__(model=model, system_prompt_override=system_prompt_override)
        self.yolo_model_path = yolo_model_path

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.detection_toolkit import YOLODetectionToolkit
            tk = YOLODetectionToolkit(yolo_model_path=self.yolo_model_path)
            tools.extend(tk.get_tools())
        except Exception:
            pass
        return tools

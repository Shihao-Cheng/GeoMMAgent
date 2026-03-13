# exec_agents/perception/cls_agent.py
# 场景分类 Agent — YOLO11-cls + VLM 推理

from typing import List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import CLS_SYSTEM_PROMPT, CLS_WORKER_DESC


class ClsAgent(BaseExecAgent):
    """
    遥感场景分类 Agent。

    使用 YOLO11 (Million-AID 51 类) 做初步分类，
    再结合 VLM 对图文进行推理，输出最终选项。
    """

    SYSTEM_PROMPT = CLS_SYSTEM_PROMPT
    WORKER_DESCRIPTION = CLS_WORKER_DESC

    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
        system_prompt_override: Optional[str] = None,
        yolo_model_path: str = "weights/yolo11s-cls.pt",
    ):
        super().__init__(model=model, system_prompt_override=system_prompt_override)
        self.yolo_model_path = yolo_model_path

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.classification_toolkit import ClassificationToolkit
            tk = ClassificationToolkit(yolo_model_path=self.yolo_model_path)
            tools.extend(tk.get_tools())
        except Exception:
            pass
        return tools

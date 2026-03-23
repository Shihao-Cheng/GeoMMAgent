# exec_agents/perception/cls_agent.py
# 场景分类 Agent — 支持多 YOLO 权重，LLM 根据任务选择调用哪个

import pathlib
from typing import Dict, List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import CLS_SYSTEM_PROMPT, CLS_WORKER_DESC

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


class ClsAgent(BaseExecAgent):
    """
    遥感场景分类 Agent。

    支持注册多个 YOLO 分类权重，每个权重生成一个独立的分类工具，
    LLM 根据 subtask 内容选择最合适的工具调用。
    """

    SYSTEM_PROMPT = CLS_SYSTEM_PROMPT
    WORKER_DESCRIPTION = CLS_WORKER_DESC

    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
        system_prompt_override: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
        yolo_weights: Optional[List[Dict]] = None,
    ):
        super().__init__(model=model, system_prompt_override=system_prompt_override)
        if yolo_weights:
            self.yolo_weights = yolo_weights
        elif yolo_model_path:
            self.yolo_weights = [{
                "path": yolo_model_path,
                "name": "classify_scene",
                "description": "General scene classification",
            }]
        else:
            self.yolo_weights = [{
                "path": str(_PROJECT_ROOT / "weights" / "yolo26x-cls.pt"),
                "name": "classify_scene",
                "description": "General scene classification",
            }]

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.classification_toolkit import ClassificationToolkit
            for w in self.yolo_weights:
                path = w["path"]
                if not pathlib.Path(path).is_absolute():
                    path = str(_PROJECT_ROOT / path)
                tk = ClassificationToolkit(yolo_model_path=path)
                func = tk.classify_image_from_path
                func.__name__ = w.get("name", "classify_scene")
                func.__doc__ = w.get("description", "Classify scene in image")
                tools.append(FunctionTool(func))
        except Exception:
            pass
        return tools

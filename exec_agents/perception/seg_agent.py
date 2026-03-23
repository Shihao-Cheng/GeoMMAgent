# exec_agents/perception/seg_agent.py
# 语义分割 Agent — DeepLabV3+（Xception），按类别统计像素面积

import pathlib
from typing import Dict, List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import SEG_SYSTEM_PROMPT, SEG_WORKER_DESC

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


class SegAgent(BaseExecAgent):
    """
    遥感语义分割 Agent。

    使用 DeepLabV3+（Xception 骨干）权重，按预测类别汇总像素面积与占整图比例。
    """

    SYSTEM_PROMPT = SEG_SYSTEM_PROMPT
    WORKER_DESCRIPTION = SEG_WORKER_DESC

    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
        system_prompt_override: Optional[str] = None,
        deeplab_weights: Optional[List[Dict]] = None,
    ):
        super().__init__(model=model, system_prompt_override=system_prompt_override)
        if deeplab_weights:
            self.deeplab_weights = deeplab_weights
        else:
            self.deeplab_weights = [{
                "path": str(_PROJECT_ROOT / "weights" / "deeplabv3plus-loveda.pth"),
                "name": "segment_image",
                "num_classes": 7,
                "output_stride": 16,
                "description": (
                    "Semantic segmentation (DeepLabV3+ Xception); per-class "
                    "pixel area and ratio for land-cover / regional statistics."
                ),
            }]

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.segmentation_toolkit import SegmentationToolkit
            for w in self.deeplab_weights:
                path = w["path"]
                if not pathlib.Path(path).is_absolute():
                    path = str(_PROJECT_ROOT / path)
                tk = SegmentationToolkit(
                    checkpoint_path=path,
                    num_classes=int(w.get("num_classes", 7)),
                    output_stride=int(w.get("output_stride", 16)),
                )
                func = tk.segment_image_from_path
                func.__name__ = w.get("name", "segment_image")
                func.__doc__ = w.get(
                    "description",
                    "Segment image and return per-class area statistics.",
                )
                tools.append(FunctionTool(func))
        except Exception:
            pass
        return tools

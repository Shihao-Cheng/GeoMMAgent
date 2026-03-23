# toolkit/segmentation_toolkit.py
# DeepLabV3+（Xception）语义分割 → 按类别统计像素面积与占比

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import MCPServer, dependencies_required
from PIL import Image
from torchvision import transforms as T

from toolkit.perception_io import (
    TOOL_SEGMENTATION,
    wrap_err,
    wrap_ok,
)


def _load_state_into_model(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise ValueError("checkpoint format not recognized")

    stripped = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        stripped[nk] = v
    model.load_state_dict(stripped, strict=False)


@MCPServer()
class SegmentationToolkit(BaseToolkit):
    """DeepLabV3+ Xception：语义分割，按类汇总像素面积。"""

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 7,
        output_stride: int = 16,
        temp_img_dir: str = "temp_images",
    ):
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.temp_img_dir = temp_img_dir
        self._model: Optional[nn.Module] = None
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        os.makedirs(self.temp_img_dir, exist_ok=True)

    def _get_model(self) -> nn.Module:
        if self._model is not None:
            return self._model
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"DeepLab checkpoint not found: {self.checkpoint_path}"
            )
        from toolkit.deeplabv3plus_xception import deeplabv3plus_xception

        model = deeplabv3plus_xception(
            num_classes=self.num_classes,
            output_stride=self.output_stride,
            pretrained_backbone=False,
        )
        _load_state_into_model(model, self.checkpoint_path)
        model = model.to(self._device)
        model.eval()
        self._model = model
        return model

    @dependencies_required("torch", "torchvision")
    def segment_image_from_path(self, image_path: str) -> str:
        """
        对遥感图像做语义分割，按类别汇总像素数与占整幅图比例。

        Args:
            image_path: 图像文件路径。

        Returns:
            str: 统一感知外壳 JSON（ok, tool=segmentation, data.per_class 等）。
        """
        try:
            model = self._get_model()
        except Exception as e:
            return wrap_err(
                TOOL_SEGMENTATION,
                f"DeepLab 模型加载失败: {e}",
                code="LOAD_FAILED",
            )

        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            inp = transform(img).unsqueeze(0).to(self._device)
        except Exception as e:
            return wrap_err(TOOL_SEGMENTATION, f"读图失败: {e}", code="IO_ERROR")

        try:
            with torch.no_grad():
                logits = model(inp)
                pred = logits.argmax(dim=1).cpu().numpy()[0]
        except Exception as e:
            return wrap_err(
                TOOL_SEGMENTATION,
                f"DeepLab 推理失败: {e}",
                code="INFERENCE_FAILED",
            )

        total_pixels = float(pred.size)
        ids, counts = np.unique(pred, return_counts=True)
        per_class: List[Dict[str, Any]] = []
        for cid, cnt in zip(ids.tolist(), counts.tolist()):
            c = int(cid)
            px = int(cnt)
            per_class.append({
                "class_id": c,
                "area_pixels": px,
                "ratio_of_image": round(px / total_pixels, 6) if total_pixels else 0.0,
            })

        data = {
            "image_size": [w, h],
            "total_pixels": int(total_pixels),
            "per_class": per_class,
        }
        meta = {
            "checkpoint": self.checkpoint_path,
            "model_family": "deeplabv3plus_xception",
            "num_classes": self.num_classes,
            "output_stride": self.output_stride,
        }
        return wrap_ok(TOOL_SEGMENTATION, data, meta)

    def get_tools(self) -> List[FunctionTool]:
        return [FunctionTool(self.segment_image_from_path)]

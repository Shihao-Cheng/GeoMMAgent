# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import os
import uuid
from typing import Any, Dict, List
from collections import Counter, defaultdict

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import MCPServer, dependencies_required
from ultralytics import YOLO

from toolkit.perception_io import (
    TOOL_DETECTION,
    wrap_err,
    wrap_ok,
)


@MCPServer()
class YOLODetectionToolkit(BaseToolkit):
    r"""YOLO目标检测工具包
    
    该工具包提供YOLO模型的目标检测功能，可以：
    - 检测图像中的物体
    - 返回检测到的物体类别和数量
    - 支持不同的YOLO模型
    - 可配置检测参数
    """

    def __init__(
        self,
        yolo_model_path: str = "weights/yolo26x-obb.pt",
        temp_img_dir: str = "temp_images",
        yolo_output_path: str = "yolo_out",
        conf_threshold: float = 0.5,
        imgsz: int = 1280,
    ):
        """
        初始化YOLO工具包
        
        Args:
            yolo_model_path (str): YOLO模型文件路径
            temp_img_dir (str): 临时图像存储目录
            yolo_output_path (str): YOLO检测结果输出目录
            conf_threshold (float): 置信度阈值
            imgsz (int): 输入图像尺寸
        """
        self.yolo_model_path = yolo_model_path
        self.temp_img_dir = temp_img_dir
        self.yolo_output_path = yolo_output_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self._yolo_model = None
        
        # 确保必要目录存在
        os.makedirs(self.temp_img_dir, exist_ok=True)
        os.makedirs(self.yolo_output_path, exist_ok=True)

    def _get_yolo_model(self):
        """获取或创建YOLO模型实例"""
        if self._yolo_model is None:
            try:
                self._yolo_model = YOLO(self.yolo_model_path)
            except Exception as e:
                print(f"❌ YOLO 模型加载失败: {e}")
                raise e
        return self._yolo_model

    def _detections_list_from_result(self, result) -> List[Dict[str, Any]]:
        """Serialize each instance: class, confidence, geometry (no drawing). OBB: xywhr; HBB: xyxy.

        Not a ``@staticmethod``: ``BaseToolkit`` metaclass wraps callables with ``with_timeout``,
        which breaks staticmethod binding (``self`` + ``result`` would be passed to a one-arg func).
        """
        out: List[Dict[str, Any]] = []
        names = result.names
        if hasattr(result, "obb") and result.obb is not None:
            obb = result.obb
            n = len(obb.cls)
            for j in range(n):
                cls_id = int(obb.cls[j].item())
                conf = None
                if obb.conf is not None and j < len(obb.conf):
                    conf = round(float(obb.conf[j].item()), 4)
                item: Dict[str, Any] = {
                    "class_name": names[cls_id],
                    "class_id": cls_id,
                    "confidence": conf,
                }
                if getattr(obb, "xywhr", None) is not None and j < len(obb.xywhr):
                    row = obb.xywhr[j].detach().cpu().tolist()
                    item["obb_xywhr"] = [round(float(x), 4) for x in row]
                elif getattr(obb, "xyxyxyxy", None) is not None and j < len(obb.xyxyxyxy):
                    flat = obb.xyxyxyxy[j].detach().cpu().flatten().tolist()
                    item["obb_xyxyxyxy"] = [round(float(x), 2) for x in flat]
                out.append(item)
            return out
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes
            for j in range(len(boxes.cls)):
                cls_id = int(boxes.cls[j].item())
                conf = None
                if boxes.conf is not None and j < len(boxes.conf):
                    conf = round(float(boxes.conf[j].item()), 4)
                item = {
                    "class_name": names[cls_id],
                    "class_id": cls_id,
                    "confidence": conf,
                }
                if getattr(boxes, "xyxy", None) is not None and j < len(boxes.xyxy):
                    xyxy = boxes.xyxy[j].detach().cpu().tolist()
                    item["xyxy"] = [round(float(x), 2) for x in xyxy]
                out.append(item)
        return out

    @dependencies_required("ultralytics")
    def detect_objects_from_image_path(self, image_path: str) -> str:
        """
        从图像文件路径检测物体
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            str: 统一感知外壳 JSON（ok, tool=detection；data 含 ``detections`` 坐标、
            per_class_counts 等；不生成带框可视化图）。
        """
        try:
            yolo_model = self._get_yolo_model()
            # Inference tensors do not require drawing; do not pass line_width=float (Ultralytics requires int).
            results = yolo_model(
                image_path,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                save=False,
                verbose=False,
            )

            detections_list: List[Dict[str, Any]] = []
            class_counts = Counter()
            conf_sum = defaultdict(float)
            image_size = None
            for i, result in enumerate(results):
                detections_list.extend(self._detections_list_from_result(result))
                if getattr(result, "orig_shape", None) is not None:
                    oh, ow = result.orig_shape
                    image_size = [int(ow), int(oh)]
                if hasattr(result, 'obb') and result.obb is not None:
                    obb = result.obb
                    for j in range(len(obb.cls)):
                        name = result.names[int(obb.cls[j].item())]
                        class_counts[name] += 1
                        if obb.conf is not None and len(obb.conf) > j:
                            conf_sum[name] += float(obb.conf[j].item())
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for j in range(len(boxes.cls)):
                        name = result.names[int(boxes.cls[j].item())]
                        class_counts[name] += 1
                        if boxes.conf is not None and len(boxes.conf) > j:
                            conf_sum[name] += float(boxes.conf[j].item())

            per_class_avg_conf = {
                k: round(conf_sum[k] / class_counts[k], 4)
                for k in class_counts
                if class_counts[k] > 0
            }
            total_instances = int(sum(class_counts.values()))
            data = {
                "image_size": image_size,
                "per_class_counts": dict(class_counts),
                "per_class_avg_confidence": per_class_avg_conf,
                "total_instances": total_instances,
                "detections": detections_list,
            }
            meta = {
                "checkpoint": self.yolo_model_path,
                "model_family": "yolo_obb",
            }
            return wrap_ok(TOOL_DETECTION, data, meta)

        except Exception as e:
            error_msg = f"YOLO检测过程中出现错误: {str(e)}"
            print(error_msg)
            return wrap_err(TOOL_DETECTION, error_msg, code="INFERENCE_FAILED")

    @dependencies_required("ultralytics")
    def detect_objects_from_image_bytes(self, image_bytes: bytes) -> str:
        """
        从图像字节数据检测物体
        
        Args:
            image_bytes (bytes): 图像的字节数据
            
        Returns:
            str: 与 detect_objects_from_image_path 相同的外壳 JSON。
        """
        temp_yolo_img_path = os.path.join(self.temp_img_dir, f"yolo_temp_{uuid.uuid4()}.png")
        
        try:
            # 保存临时图像文件
            with open(temp_yolo_img_path, 'wb') as f:
                f.write(image_bytes)
            
            # 使用图像路径进行检测
            result = self.detect_objects_from_image_path(temp_yolo_img_path)
            return result

        finally:
            # 清理临时文件
            if os.path.exists(temp_yolo_img_path):
                os.remove(temp_yolo_img_path)

    @dependencies_required("ultralytics")
    def detect_objects_from_pil_image(self, image_data: str) -> str:
        """
        从PIL图像对象检测物体
        
        Args:
            image_data (str): 图像的base64编码字符串或图像路径
            
        Returns:
            str: 与 detect_objects_from_image_path 相同的外壳 JSON。
        """
        try:
            # 如果是文件路径，直接使用
            if os.path.isfile(image_data):
                return self.detect_objects_from_image_path(image_data)
            
            # 如果是base64编码的图像数据
            import base64
            if image_data.startswith('data:image'):
                # 处理data URL格式
                header, encoded = image_data.split(',', 1)
                image_bytes = base64.b64decode(encoded)
            else:
                # 直接是base64编码
                image_bytes = base64.b64decode(image_data)
            
            return self.detect_objects_from_image_bytes(image_bytes)
            
        except Exception as e:
            # 如果上述方法都失败，尝试作为路径处理
            if os.path.isfile(image_data):
                return self.detect_objects_from_image_path(image_data)
            
            error_msg = f"无法处理图像数据: {str(e)}"
            print(error_msg)
            return wrap_err(TOOL_DETECTION, error_msg, code="IO_ERROR")

    def get_tools(self) -> List[FunctionTool]:
        r"""返回YOLO工具包中的所有工具
        
        Returns:
            List[FunctionTool]: 工具函数列表
        """
        return [
            FunctionTool(self.detect_objects_from_image_path),
            FunctionTool(self.detect_objects_from_image_bytes),
            FunctionTool(self.detect_objects_from_pil_image),
        ]

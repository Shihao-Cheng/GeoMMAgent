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
import io
import base64
from typing import List

from PIL import Image

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import MCPServer, dependencies_required
from ultralytics import YOLO

from toolkit.perception_io import (
    TOOL_CLASSIFICATION,
    wrap_err,
    wrap_ok,
)


@MCPServer()
class ClassificationToolkit(BaseToolkit):
    r"""统一图像分类工具包
    
    该工具包提供多种图像分类功能：
    - 支持多种输入格式（文件路径、字节数据、PIL图像）
    - 返回 top-1 和 top-5 预测类别及置信度
    """

    def __init__(
        self,
        yolo_model_path: str = "weights/yolo26x-cls.pt",
        temp_img_dir: str = "temp_images",
        yolo_output_path: str = "yolo_out",
        conf_threshold: float = 0.5,
        imgsz: int = 224,
    ):
        self.yolo_model_path = yolo_model_path
        self.temp_img_dir = temp_img_dir
        self.yolo_output_path = yolo_output_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self._yolo_model = None

        os.makedirs(self.temp_img_dir, exist_ok=True)
        os.makedirs(self.yolo_output_path, exist_ok=True)

    def _get_yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = YOLO(self.yolo_model_path)
        return self._yolo_model

    @dependencies_required("ultralytics")
    def classify_image_from_path(self, image_path: str) -> str:
        """
        从图像文件路径进行分类预测
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            str: 统一感知外壳 JSON（schema_version, ok, tool=classification, data, meta）。
        """
        try:
            yolo_model = self._get_yolo_model()
            results = yolo_model(
                image_path,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                save=True,
                project=self.yolo_output_path,
                exist_ok=True,
                name="yolo_classification_results"
            )
            
            classification_results = []
            image_size = None
            for i, result in enumerate(results):
                if getattr(result, "orig_shape", None) is not None:
                    oh, ow = result.orig_shape
                    image_size = [int(ow), int(oh)]
                if hasattr(result, 'probs') and result.probs is not None:
                    # 获取分类结果
                    probs = result.probs
                    top_class_idx = probs.top1
                    top_class_name = result.names[top_class_idx]
                    top_confidence = probs.top1conf.item()
                    
                    # 获取前5个预测结果
                    top5_indices = probs.top5
                    top5_results = []
                    for idx in top5_indices:
                        class_name = result.names[idx]
                        confidence = probs.data[idx].item()
                        top5_results.append({
                            "class": class_name,
                            "confidence": round(confidence, 4)
                        })
                    
                    classification_result = {
                        "top_prediction": {
                            "class": top_class_name,
                            "confidence": round(top_confidence, 4)
                        },
                        "top5_predictions": top5_results
                    }
                    classification_results.append(classification_result)
                else:
                    classification_results.append({
                        "error": "未找到分类结果"
                    })
            
            data = {
                "image_size": image_size,
                "predictions": classification_results,
            }
            meta = {
                "checkpoint": self.yolo_model_path,
                "model_family": "yolo_cls",
            }
            print(f"YOLO分类结果: {data}")
            return wrap_ok(TOOL_CLASSIFICATION, data, meta)

        except Exception as e:
            error_msg = f"YOLO分类过程中出现错误: {str(e)}"
            print(error_msg)
            return wrap_err(TOOL_CLASSIFICATION, error_msg, code="INFERENCE_FAILED")

    @dependencies_required("ultralytics")
    def classify_image_from_bytes(self, image_bytes: bytes) -> str:
        """
        从图像字节数据进行分类预测
        
        Args:
            image_bytes (bytes): 图像的字节数据
            
        Returns:
            str: 与 classify_image_from_path 相同的外壳 JSON。
        """
        temp_yolo_img_path = os.path.join(self.temp_img_dir, f"yolo_cls_temp_{uuid.uuid4()}.png")
        
        try:
            # 保存临时图像文件
            with open(temp_yolo_img_path, 'wb') as f:
                f.write(image_bytes)
            
            # 使用图像路径进行分类
            result = self.classify_image_from_path(temp_yolo_img_path)
            return result

        finally:
            # 清理临时文件
            if os.path.exists(temp_yolo_img_path):
                os.remove(temp_yolo_img_path)

    @dependencies_required("ultralytics")
    def classify_image_from_pil_image(self, image_data: str) -> str:
        """
        从PIL图像对象进行分类预测
        
        Args:
            image_data (str): 图像的base64编码字符串或图像路径
            
        Returns:
            str: 与 classify_image_from_path 相同的外壳 JSON。
        """
        try:
            # 如果是文件路径，直接使用
            if os.path.isfile(image_data):
                return self.classify_image_from_path(image_data)
            
            # 如果是base64编码的图像数据
            if image_data.startswith('data:image'):
                # 处理data URL格式
                header, encoded = image_data.split(',', 1)
                image_bytes = base64.b64decode(encoded)
            else:
                # 直接是base64编码
                image_bytes = base64.b64decode(image_data)
            
            return self.classify_image_from_bytes(image_bytes)
            
        except Exception as e:
            # 如果上述方法都失败，尝试作为路径处理
            if os.path.isfile(image_data):
                return self.classify_image_from_path(image_data)
            
            error_msg = f"无法处理图像数据: {str(e)}"
            print(error_msg)
            return wrap_err(TOOL_CLASSIFICATION, error_msg, code="IO_ERROR")

    def get_tools(self) -> List[FunctionTool]:
        r"""返回分类工具包中的所有工具
        
        Returns:
            List[FunctionTool]: 工具函数列表
        """
        return [
            FunctionTool(self.classify_image_from_path),
            FunctionTool(self.classify_image_from_bytes),
            FunctionTool(self.classify_image_from_pil_image),
        ]

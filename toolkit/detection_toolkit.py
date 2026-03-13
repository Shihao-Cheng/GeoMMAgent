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
import json
from typing import Dict, Any, Optional, List
from collections import Counter
from PIL import Image
import io

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import dependencies_required
from ultralytics import YOLO


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
        yolo_model_path: str = "/root/autodl-tmp/yolo11s-obb.pt",
        temp_img_dir: str = "temp_images",
        yolo_output_path: str = "/root/autodl-tmp/yolo_out/",
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
            print("🚀 正在加载 YOLO 模型...")
            try:
                self._yolo_model = YOLO(self.yolo_model_path)
                print("✅ YOLO 模型加载成功！")
            except Exception as e:
                print(f"❌ YOLO 模型加载失败: {e}")
                raise e
        return self._yolo_model

    @dependencies_required("ultralytics")
    def detect_objects_from_image_path(self, image_path: str) -> str:
        """
        从图像文件路径检测物体
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            str: JSON格式的检测结果，包含各类物体的数量统计
        """
        try:
            yolo_model = self._get_yolo_model()
            results = yolo_model(
                image_path,
                line_width=1,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                save=True,
                project=self.yolo_output_path,
                exist_ok=True,
                name="yolo_results"
            )
            
            class_counts = Counter()
            for i, result in enumerate(results):
                if hasattr(result, 'obb') and result.obb is not None:
                    # 有向边界框检测 (OBB)
                    names = [result.names[cls.item()] for cls in result.obb.cls.int()]
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    # 普通边界框检测
                    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
                else:
                    names = []
                
                for name in names:
                    class_counts[name] += 1

            detection_result = dict(class_counts)
            print(f"YOLO检测到的所有类别和数量: {detection_result}")
            
            return json.dumps(detection_result, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"YOLO检测过程中出现错误: {str(e)}"
            print(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)

    @dependencies_required("ultralytics")
    def detect_objects_from_image_bytes(self, image_bytes: bytes) -> str:
        """
        从图像字节数据检测物体
        
        Args:
            image_bytes (bytes): 图像的字节数据
            
        Returns:
            str: JSON格式的检测结果，包含各类物体的数量统计
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
            str: JSON格式的检测结果，包含各类物体的数量统计
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
            return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)

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

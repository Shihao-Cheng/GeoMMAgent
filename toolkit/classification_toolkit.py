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
import sys
from typing import Dict, Any, Optional, List
from PIL import Image
import io
import base64

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import dependencies_required
from ultralytics import YOLO

# 添加路径以导入 RSClassificationAgent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



class ClassificationToolkit(BaseToolkit):
    r"""统一图像分类工具包
    
    该工具包提供多种图像分类功能：
    - 纯YOLO分类：快速、准确的图像分类
    - 遥感图像分类：结合YOLO和Qwen视觉模型的智能分析
    - 支持多种输入格式（文件路径、字节数据、PIL图像）
    - 提供详细的分类结果和置信度
    - 支持选项过滤功能
    """

    def __init__(
        self,
        yolo_model_path: str = "/root/autodl-tmp/yolo11s-cls.pt",
        temp_img_dir: str = "temp_images",
        yolo_output_path: str = "/root/autodl-tmp/yolo_out/",
        conf_threshold: float = 0.5,
        imgsz: int = 224,
        request_delay: float = 0.2,
        max_workers: int = 20,
        model_config: Optional[Dict] = None,
        enable_rs_analysis: bool = True
    ):
        """
        初始化分类工具包
        
        Args:
            yolo_model_path (str): YOLO分类模型文件路径
            temp_img_dir (str): 临时图像存储目录
            yolo_output_path (str): YOLO分类结果输出目录
            conf_threshold (float): 置信度阈值
            imgsz (int): 输入图像尺寸 (分类模型通常使用224)
            request_delay (float): 请求间延迟（秒）
            max_workers (int): 最大工作线程数
            model_config (dict, optional): 模型配置
            enable_rs_analysis (bool): 是否启用遥感分析功能
        """
        self.yolo_model_path = yolo_model_path
        self.temp_img_dir = temp_img_dir
        self.yolo_output_path = yolo_output_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.request_delay = request_delay
        self.max_workers = max_workers
        self.model_config = model_config
        self.enable_rs_analysis = enable_rs_analysis
        self._yolo_model = None
        self._rs_agent = None
        
        # 确保必要目录存在
        os.makedirs(self.temp_img_dir, exist_ok=True)
        os.makedirs(self.yolo_output_path, exist_ok=True)

    def _get_yolo_model(self):
        """获取或创建YOLO分类模型实例"""
        if self._yolo_model is None:
            print("🚀 正在加载 YOLO 分类模型...")
            try:
                self._yolo_model = YOLO(self.yolo_model_path)
                print("✅ YOLO 分类模型加载成功！")
            except Exception as e:
                print(f"❌ YOLO 分类模型加载失败: {e}")
                raise e
        return self._yolo_model

    def _get_rs_agent(self):
        """获取或创建RSClassificationAgent实例"""
        if self._rs_agent is None and self.enable_rs_analysis:
            print("🚀 正在初始化遥感分类代理...")
            try:
                # 延迟导入以避免循环导入
                try:
                    from agent.classifier import RSClassificationAgent
                except ImportError:
                    from classifier import RSClassificationAgent
                
                self._rs_agent = RSClassificationAgent(
                    yolo_model_path=self.yolo_model_path,
                    temp_img_dir=self.temp_img_dir,
                    yolo_output_path=self.yolo_output_path,
                    request_delay=self.request_delay,
                    max_workers=self.max_workers,
                    model_config=self.model_config
                )
                print("✅ 遥感分类代理初始化成功！")
            except Exception as e:
                print(f"❌ 遥感分类代理初始化失败: {e}")
                raise e
        return self._rs_agent

    @dependencies_required("ultralytics")
    def classify_image_from_path(self, image_path: str) -> str:
        """
        从图像文件路径进行分类预测
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            str: JSON格式的分类结果，包含预测类别和置信度
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
            for i, result in enumerate(results):
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
            
            final_result = {
                "classification_results": classification_results,
                "model_path": self.yolo_model_path
            }
            
            print(f"YOLO分类结果: {final_result}")
            return json.dumps(final_result, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"YOLO分类过程中出现错误: {str(e)}"
            print(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)

    @dependencies_required("ultralytics")
    def classify_image_from_bytes(self, image_bytes: bytes) -> str:
        """
        从图像字节数据进行分类预测
        
        Args:
            image_bytes (bytes): 图像的字节数据
            
        Returns:
            str: JSON格式的分类结果，包含预测类别和置信度
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
            str: JSON格式的分类结果，包含预测类别和置信度
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
            return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)

    def filter_classifications_by_options(self, classification_results: List[Dict], options: Dict[str, str]) -> List[Dict]:
        """
        根据选项过滤分类结果，只保留选项中存在的分类
        
        Args:
            classification_results (List[Dict]): YOLO分类结果列表
            options (Dict[str, str]): 选项字典，格式为 {label: text}
            
        Returns:
            List[Dict]: 过滤后的分类结果
        """
        if not classification_results or not options:
            return classification_results
        
        # 提取所有选项文本
        option_texts = [text.lower() for text in options.values() if text]
        print(f"Available option texts: {option_texts}")
        
        filtered_results = []
        for result in classification_results:
            if "top5_predictions" in result:
                # 过滤top5预测结果
                filtered_top5 = []
                for prediction in result["top5_predictions"]:
                    pred_class = prediction["class"].lower()
                    
                    # 检查预测类别是否与任何选项匹配
                    is_match = False
                    for option_text in option_texts:
                        if (pred_class in option_text) or (option_text in pred_class):
                            is_match = True
                            print(f"Keeping prediction: {pred_class} (matches option: {option_text})")
                            break
                    
                    if is_match:
                        filtered_top5.append(prediction)
                    else:
                        print(f"Filtering out prediction: {pred_class} (no matching option)")
                
                # 更新结果
                result_copy = result.copy()
                result_copy["top5_predictions"] = filtered_top5
                
                # 如果top1预测被过滤掉了，更新top1为过滤后列表的第一个
                if filtered_top5:
                    result_copy["top_prediction"] = filtered_top5[0]
                else:
                    # 如果没有匹配的预测，保持原始结果但标记为无匹配
                    result_copy["no_matching_options"] = True
                
                filtered_results.append(result_copy)
            else:
                # 如果没有top5预测，保持原结果
                filtered_results.append(result)
        
        return filtered_results

    @dependencies_required("ultralytics")
    def classify_image_with_options_filter(self, image_path: str, options: Dict[str, str]) -> str:
        """
        从图像文件路径进行分类预测，并过滤掉不在选项中的分类结果
        
        Args:
            image_path (str): 图像文件路径
            options (Dict[str, str]): 选项字典，格式为 {label: text}
            
        Returns:
            str: JSON格式的过滤后分类结果
        """
        try:
            # 先进行正常分类
            raw_result = self.classify_image_from_path(image_path)
            raw_data = json.loads(raw_result)
            
            # 如果有错误，直接返回
            if "error" in raw_data:
                return raw_result
            
            # 应用选项过滤
            if "classification_results" in raw_data:
                filtered_results = self.filter_classifications_by_options(
                    raw_data["classification_results"], options
                )
                raw_data["classification_results"] = filtered_results
                raw_data["filtered_by_options"] = True
                raw_data["available_options"] = list(options.values())
            
            print(f"Filtered YOLO classification result: {raw_data}")
            return json.dumps(raw_data, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"YOLO分类过滤过程中出现错误: {str(e)}"
            print(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)

    @dependencies_required("ultralytics")
    def classify_remote_sensing_image_from_path(self, image_path: str, question: str = "请分析这张遥感图像的内容", 
                                               options: Optional[Dict[str, str]] = None) -> str:
        """
        从图像文件路径进行遥感图像分类分析
        
        Args:
            image_path (str): 图像文件路径
            question (str): 分析问题
            options (dict, optional): 选项字典，格式如 {"A": "选项A", "B": "选项B"}
            
        Returns:
            str: JSON格式的分类分析结果
        """
        if not self.enable_rs_analysis:
            return json.dumps({"error": "遥感分析功能未启用"}, ensure_ascii=False, indent=2)
            
        try:
            # 读取图像文件
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            return self.classify_remote_sensing_image_from_bytes(image_bytes, question, options)
            
        except Exception as e:
            error_msg = f"读取图像文件失败: {str(e)}"
            print(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)

    @dependencies_required("ultralytics")
    def classify_remote_sensing_image_from_bytes(self, image_bytes: bytes, question: str = "请分析这张遥感图像的内容",
                                                options: Optional[Dict[str, str]] = None) -> str:
        """
        从图像字节数据进行遥感图像分类分析
        
        Args:
            image_bytes (bytes): 图像的字节数据
            question (str): 分析问题
            options (dict, optional): 选项字典，格式如 {"A": "选项A", "B": "选项B"}
            
        Returns:
            str: JSON格式的分类分析结果
        """
        if not self.enable_rs_analysis:
            return json.dumps({"error": "遥感分析功能未启用"}, ensure_ascii=False, indent=2)
            
        try:
            agent = self._get_rs_agent()
            
            # 创建模拟查询数据
            query_data = {
                'index': 0,
                'image': {'bytes': image_bytes},
                'question': question,
                'A': options.get('A', '') if options else '',
                'B': options.get('B', '') if options else '',
                'C': options.get('C', '') if options else '',
                'D': options.get('D', '') if options else '',
                'answer': 'N/A'  # 用于测试，实际使用时需要提供正确答案
            }
            
            # 处理单个查询
            result = agent.process_single_query(query_data)
            
            # 格式化结果
            formatted_result = {
                "question": result.get('question', ''),
                "yolo_classification": result.get('yolo_classification_result', {}),
                "ai_prediction": result.get('pred_qwen', ''),
                "raw_response": result.get('raw_response', ''),
                "analysis_summary": self._create_analysis_summary(result)
            }
            
            return json.dumps(formatted_result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            error_msg = f"遥感图像分类分析失败: {str(e)}"
            print(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)

    @dependencies_required("ultralytics")
    def classify_remote_sensing_image_from_pil_image(self, image_data: str, question: str = "请分析这张遥感图像的内容",
                                                    options: Optional[Dict[str, str]] = None) -> str:
        """
        从PIL图像对象进行遥感图像分类分析
        
        Args:
            image_data (str): 图像的base64编码字符串或图像路径
            question (str): 分析问题
            options (dict, optional): 选项字典，格式如 {"A": "选项A", "B": "选项B"}
            
        Returns:
            str: JSON格式的分类分析结果
        """
        if not self.enable_rs_analysis:
            return json.dumps({"error": "遥感分析功能未启用"}, ensure_ascii=False, indent=2)
            
        try:
            # 如果是文件路径，直接使用
            if os.path.isfile(image_data):
                return self.classify_remote_sensing_image_from_path(image_data, question, options)
            
            # 如果是base64编码的图像数据
            if image_data.startswith('data:image'):
                # 处理data URL格式
                header, encoded = image_data.split(',', 1)
                image_bytes = base64.b64decode(encoded)
            else:
                # 直接是base64编码
                image_bytes = base64.b64decode(image_data)
            
            return self.classify_remote_sensing_image_from_bytes(image_bytes, question, options)
            
        except Exception as e:
            # 如果上述方法都失败，尝试作为路径处理
            if os.path.isfile(image_data):
                return self.classify_remote_sensing_image_from_path(image_data, question, options)
            
            error_msg = f"无法处理图像数据: {str(e)}"
            print(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)

    def _create_analysis_summary(self, result: Dict[str, Any]) -> str:
        """创建分析摘要"""
        try:
            yolo_result = result.get('yolo_classification_result', {})
            ai_prediction = result.get('pred_qwen', '')
            raw_response = result.get('raw_response', '')
            
            summary_parts = []
            
            # YOLO分类结果
            if yolo_result and "classification_results" in yolo_result:
                classification_results = yolo_result["classification_results"]
                if classification_results and len(classification_results) > 0:
                    top_pred = classification_results[0].get("top_prediction", {})
                    if top_pred:
                        summary_parts.append(f"YOLO分类结果: {top_pred.get('class', 'N/A')} (置信度: {top_pred.get('confidence', 0):.2f})")
            
            # AI预测结果
            if ai_prediction and ai_prediction != "UNKNOWN":
                summary_parts.append(f"AI预测结果: {ai_prediction}")
            
            # 原始响应摘要
            if raw_response and len(raw_response) > 50:
                summary_parts.append(f"详细分析: {raw_response[:100]}...")
            
            return " | ".join(summary_parts) if summary_parts else "分析完成，但未获得具体结果"
            
        except Exception as e:
            return f"分析摘要生成失败: {str(e)}"

    def get_tools(self) -> List[FunctionTool]:
        r"""返回分类工具包中的所有工具
        
        Returns:
            List[FunctionTool]: 工具函数列表
        """
        tools = [
            FunctionTool(self.classify_image_from_path),
            FunctionTool(self.classify_image_from_bytes),
            FunctionTool(self.classify_image_from_pil_image),
            FunctionTool(self.classify_image_with_options_filter),
        ]
        
        # 如果启用了遥感分析功能，添加相关工具
        if self.enable_rs_analysis:
            tools.extend([
                FunctionTool(self.classify_remote_sensing_image_from_path),
                FunctionTool(self.classify_remote_sensing_image_from_bytes),
                FunctionTool(self.classify_remote_sensing_image_from_pil_image),
            ])
        
        return tools

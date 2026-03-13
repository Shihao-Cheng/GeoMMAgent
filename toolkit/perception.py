# toolkit/perception.py
# 感知工具：场景分类、目标检测、语义分割
# 目前为占位实现，替换为真实模型调用即可

from typing import List
from camel.toolkits import FunctionTool


def classify_scene(image_path: str) -> str:
    """
    对遥感图像进行场景分类。

    参数:
        image_path (str): 图像路径或 URL。

    返回:
        str: 分类结果，格式为 'class_label (confidence%)'。
    """
    # TODO: 接入 YOLO11 分类模型
    return f"[classify_scene] placeholder — image: {image_path}"


def detect_objects(image_path: str) -> str:
    """
    对遥感图像进行有向目标检测。

    参数:
        image_path (str): 图像路径或 URL。

    返回:
        str: 检测结果，包含类别、数量和置信度。
    """
    # TODO: 接入 YOLO11 DOTA-v2 检测模型
    return f"[detect_objects] placeholder — image: {image_path}"


def segment_image(image_path: str) -> str:
    """
    对遥感图像进行像素级语义分割。

    参数:
        image_path (str): 图像路径或 URL。

    返回:
        str: 分割结果摘要，包含各类别面积占比。
    """
    # TODO: 接入 DeepLabv3+ LoveDA 分割模型
    return f"[segment_image] placeholder — image: {image_path}"


def get_perception_tools() -> List[FunctionTool]:
    """返回感知工具列表，供 PerceptionAgent 使用"""
    return [
        FunctionTool(classify_scene),
        FunctionTool(detect_objects),
        FunctionTool(segment_image),
    ]

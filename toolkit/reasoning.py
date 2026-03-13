# toolkit/reasoning.py
# 推理工具：空间时序分析、选项匹配
# 目前为占位实现

from typing import List
from camel.toolkits import FunctionTool


def match_answer_to_choices(free_text_answer: str, choices: str) -> str:
    """
    将自由文本回答与多选题选项进行语义匹配，返回最优选项。

    参数:
        free_text_answer (str): 推理得出的自由文本答案。
        choices (str): 选项字符串，格式如 'A. xxx\\nB. xxx\\nC. xxx\\nD. xxx'。

    返回:
        str: 最匹配的选项标识（如 'A'）及匹配依据。
    """
    # TODO: 实现语义相似度匹配逻辑
    return f"[match_answer_to_choices] placeholder — answer: {free_text_answer[:40]}"


def analyze_spatial_change(image_before: str, image_after: str) -> str:
    """
    分析两幅遥感图像之间的空间或时序变化。

    参数:
        image_before (str): 前时相图像路径。
        image_after (str): 后时相图像路径。

    返回:
        str: 变化分析结果摘要。
    """
    # TODO: 实现时序变化检测逻辑
    return f"[analyze_spatial_change] placeholder — before: {image_before}"


def get_reasoning_tools() -> List[FunctionTool]:
    """返回推理工具列表，供 ReasoningAgent 使用"""
    return [
        FunctionTool(match_answer_to_choices),
        FunctionTool(analyze_spatial_change),
    ]

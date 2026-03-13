# toolkit/knowledge.py
# RS 专用知识工具（仅保留框架无内置的部分）
#
# 已由框架覆盖，无需在此重复定义：
#   - 网页/Wikipedia 搜索 → camel.toolkits.SearchToolkit
#   - 文档/URL 向量检索   → camel.retrievers.AutoRetriever / VectorRetriever

from typing import List
from camel.toolkits import FunctionTool


def retrieve_multimodal(query: str, image_path: str = "") -> str:
    """
    使用 GME 多模态检索模型，根据文本查询和可选图像检索相关遥感知识。

    参数:
        query (str): 文本查询内容（如问题描述或关键词）。
        image_path (str): 可选的参考图像路径，用于图文联合检索。

    返回:
        str: 检索到的知识片段（含来源）。
    """
    # TODO: 接入 GME 检索模型 (zhang2025gme)
    return f"[retrieve_multimodal] placeholder — query: {query}"


def get_knowledge_tools() -> List[FunctionTool]:
    """返回 RS 专用知识工具列表，供 KnowledgeAgent 使用"""
    return [
        FunctionTool(retrieve_multimodal),
    ]

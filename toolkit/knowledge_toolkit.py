# toolkit/knowledge_toolkit.py
# MCP-exposed wrapper for ``knowledge.retrieve_multimodal`` (GME / placeholder).

from __future__ import annotations

from typing import List

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import MCPServer

from toolkit import knowledge as knowledge_mod


@MCPServer()
class KnowledgeToolkit(BaseToolkit):
    """Multimodal knowledge retrieval (same contract as ``get_knowledge_tools``)."""

    def retrieve_multimodal(self, query: str, image_path: str = "") -> str:
        return knowledge_mod.retrieve_multimodal(query, image_path)

    def get_tools(self) -> List[FunctionTool]:
        return [FunctionTool(self.retrieve_multimodal)]

# toolkit/search_toolkit.py
# MCP exposure for web search + image URL metadata + structured merge (no downloads).

from __future__ import annotations

from typing import List

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import MCPServer


@MCPServer(server_name="SearchRSToolkit")
class SearchRSToolkit(BaseToolkit):
    r"""
    - ``search_text_web``: text-first web search (DDG / Google / Bing / Wiki).
    - ``image_search_metadata``: JSON list of image URLs for downstream filtering or download.
    - ``pack_search_evidence``: merge text + ``image_search_metadata`` JSON for agents.
    """

    def search_text_web(self, query: str, max_results: int = 5) -> str:
        from exec_agents.knowledge.search_agent import force_search

        return force_search(query, max_results=max_results)

    def image_search_metadata(self, query: str, max_results: int = 5) -> str:
        from exec_agents.knowledge.search_agent import get_image_search_metadata_json

        return get_image_search_metadata_json(query, max_results=max_results)

    def pack_search_evidence(self, text_block: str, image_metadata_json: str = "") -> str:
        from exec_agents.knowledge.search_agent import pack_search_evidence_payload

        return pack_search_evidence_payload(text_block, image_metadata_json)

    def get_tools(self) -> List[FunctionTool]:
        return [
            FunctionTool(self.search_text_web),
            FunctionTool(self.image_search_metadata),
            FunctionTool(self.pack_search_evidence),
        ]

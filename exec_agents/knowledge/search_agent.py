# exec_agents/knowledge/search_agent.py
# 搜索 Agent — 基于 CAMEL SearchToolkit（DuckDuckGo / Google / Wiki / Bing）

from typing import List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool, SearchToolkit

from exec_agents.base import BaseExecAgent
from .prompts import SEARCH_SYSTEM_PROMPT, SEARCH_WORKER_DESC


class SearchAgent(BaseExecAgent):
    """
    Web 搜索 Agent。

    直接复用 CAMEL 内置 SearchToolkit，
    支持 DuckDuckGo / Google / Wikipedia / Bing。
    """

    SYSTEM_PROMPT = SEARCH_SYSTEM_PROMPT
    WORKER_DESCRIPTION = SEARCH_WORKER_DESC

    def get_tools(self) -> List[FunctionTool]:
        try:
            return SearchToolkit().get_tools()
        except Exception:
            return []

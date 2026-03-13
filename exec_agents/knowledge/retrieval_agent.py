# exec_agents/knowledge/retrieval_agent.py
# 多模态检索 Agent — GME 检索 + CAMEL AutoRetriever

from typing import List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import RETRIEVAL_SYSTEM_PROMPT, RETRIEVAL_WORKER_DESC


class RetrievalAgent(BaseExecAgent):
    """
    多模态知识检索 Agent。

    - GME 图文联合检索（RS 专用，toolkit/knowledge.py）
    - 可扩展 CAMEL AutoRetriever 做文档 RAG
    """

    SYSTEM_PROMPT = RETRIEVAL_SYSTEM_PROMPT
    WORKER_DESCRIPTION = RETRIEVAL_WORKER_DESC

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit import get_knowledge_tools
            tools.extend(get_knowledge_tools())
        except ImportError:
            pass
        return tools

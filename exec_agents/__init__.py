# exec_agents/__init__.py
# 所有执行单元 Agent 注册表
#
# 结构：每个子目录 = 一种能力，子目录内多个 Agent = 该能力的不同实现
#
#   general/      → GeneralAgent
#   perception/   → ClsAgent, DetAgent, SegAgent
#   knowledge/    → SearchAgent, RetrievalAgent
#   reasoning/    → ReasoningAgent, MatchingAgent
#   evaluation/   → SelfEvaluationAgent

from .base import BaseExecAgent

from .general import GeneralAgent, ALL_GENERAL_AGENTS
from .perception import ClsAgent, DetAgent, SegAgent, ALL_PERCEPTION_AGENTS
from .knowledge import SearchAgent, RetrievalAgent, ALL_KNOWLEDGE_AGENTS
from .reasoning import ReasoningAgent, MatchingAgent, ALL_REASONING_AGENTS
from .evaluation import SelfEvaluationAgent, ALL_EVALUATION_AGENTS

ALL_AGENTS = (
    ALL_GENERAL_AGENTS
    + ALL_PERCEPTION_AGENTS
    + ALL_KNOWLEDGE_AGENTS
    + ALL_REASONING_AGENTS
    + ALL_EVALUATION_AGENTS
)

__all__ = [
    "BaseExecAgent",
    # general
    "GeneralAgent",
    # perception
    "ClsAgent", "DetAgent", "SegAgent",
    # knowledge
    "SearchAgent", "RetrievalAgent",
    # reasoning
    "ReasoningAgent", "MatchingAgent",
    # evaluation
    "SelfEvaluationAgent",
    # registries
    "ALL_GENERAL_AGENTS", "ALL_PERCEPTION_AGENTS", "ALL_KNOWLEDGE_AGENTS",
    "ALL_REASONING_AGENTS", "ALL_EVALUATION_AGENTS",
    "ALL_AGENTS",
]

# exec_agents/evaluation/__init__.py
# 评估能力组：Self-Evaluation Agent

from .self_evaluation_agent import SelfEvaluationAgent

ALL_EVALUATION_AGENTS = [SelfEvaluationAgent]

__all__ = ["SelfEvaluationAgent", "ALL_EVALUATION_AGENTS"]

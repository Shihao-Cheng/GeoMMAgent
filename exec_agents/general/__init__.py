# exec_agents/general/__init__.py
# 通用工具能力组

from .general_agent import GeneralAgent

ALL_GENERAL_AGENTS = [GeneralAgent]

__all__ = ["GeneralAgent", "ALL_GENERAL_AGENTS"]

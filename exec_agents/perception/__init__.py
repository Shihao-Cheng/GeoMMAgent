# exec_agents/perception/__init__.py
# 感知能力组：场景分类 / 目标检测 / 语义分割

from .cls_agent import ClsAgent
from .det_agent import DetAgent
from .seg_agent import SegAgent

ALL_PERCEPTION_AGENTS = [ClsAgent, DetAgent, SegAgent]

__all__ = ["ClsAgent", "DetAgent", "SegAgent", "ALL_PERCEPTION_AGENTS"]

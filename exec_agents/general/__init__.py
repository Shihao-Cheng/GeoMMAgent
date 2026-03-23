# exec_agents/general/__init__.py
# 通用预处理：每个 Agent 绑定一类工具

from .format_conversion_agent import FormatConversionAgent
from .image_filter_agent import ImageFilterAgent
from .scale_agent import ScaleAgent
from .super_resolution_agent import SuperResolutionAgent

ALL_GENERAL_AGENTS = [
    FormatConversionAgent,
    ImageFilterAgent,
    ScaleAgent,
    SuperResolutionAgent,
]

__all__ = [
    "FormatConversionAgent",
    "ImageFilterAgent",
    "ScaleAgent",
    "SuperResolutionAgent",
    "ALL_GENERAL_AGENTS",
]

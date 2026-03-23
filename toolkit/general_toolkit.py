# toolkit/general_toolkit.py
# MCP-exposed wrapper for ``toolkit.general`` preprocessing utilities.

from __future__ import annotations

from typing import List

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import MCPServer

from toolkit import general as general_mod


@MCPServer()
class GeneralToolkit(BaseToolkit):
    """Format conversion, filtering, scaling, and super-resolution."""

    def convert_format(self, image_path: str, target_format: str = "png") -> str:
        return general_mod.convert_format(image_path, target_format)

    def filter_image(self, image_path: str, method: str = "gaussian") -> str:
        return general_mod.filter_image(image_path, method)

    def scale_image(self, image_path: str, scale_factor: float = 1.0) -> str:
        return general_mod.scale_image(image_path, scale_factor)

    def super_resolve(self, image_path: str, scale: int = 4) -> str:
        return general_mod.super_resolve(image_path, scale)

    def get_tools(self) -> List[FunctionTool]:
        return [
            FunctionTool(self.convert_format),
            FunctionTool(self.filter_image),
            FunctionTool(self.scale_image),
            FunctionTool(self.super_resolve),
        ]

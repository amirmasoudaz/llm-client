from __future__ import annotations

from agent_runtime.plugins.types import ToolsPlugin

from ..platform_tools import platform_load_funding_thread_context


class PlatformContextToolsPlugin(ToolsPlugin):
    def __init__(self):
        super().__init__(
            name="canapply_platform_context",
            version="0.1.0",
            description="Read-only CanApply platform DB context loaders.",
        )
        self.register_tool(platform_load_funding_thread_context)


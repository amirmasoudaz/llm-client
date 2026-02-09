"""
Plugin system for agent runtime.

This module provides the plugin infrastructure for extending
the runtime with:
- Tools
- Operators (workflows)
- Policies
- Connectors
- Memory providers
"""

from .types import (
    Plugin,
    PluginManifest,
    PluginType,
    PluginCapability,
    ToolsPlugin,
)
from .registry import (
    PluginRegistry,
    PluginInfo,
)

__all__ = [
    "Plugin",
    "PluginManifest",
    "PluginType",
    "PluginCapability",
    "ToolsPlugin",
    "PluginRegistry",
    "PluginInfo",
]

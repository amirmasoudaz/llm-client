"""
Plugin types for agent runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_client.tools.base import Tool


class PluginType(str, Enum):
    """Types of plugins."""
    TOOLS = "tools"              # Provides tools
    OPERATOR = "operator"        # Provides a workflow/operator
    POLICY = "policy"            # Provides policy rules
    CONNECTOR = "connector"      # Provides external connectors
    MEMORY = "memory"            # Provides memory storage
    UI = "ui"                    # Provides UI adapters


@dataclass
class PluginCapability:
    """A capability provided by a plugin."""
    name: str
    type: PluginType
    description: str = ""
    version: str = "1.0.0"
    
    # For tools: list of tool names
    tool_names: list[str] = field(default_factory=list)
    
    # For operators: operator identifier
    operator_id: str | None = None
    
    # For policies: policy names
    policy_names: list[str] = field(default_factory=list)
    
    # For connectors: connector names
    connector_names: list[str] = field(default_factory=list)


@dataclass
class PluginManifest:
    """Manifest describing a plugin.
    
    Plugins declare their capabilities via a manifest, which is used
    for discovery, dependency resolution, and capability gating.
    """
    name: str
    version: str
    description: str = ""
    author: str = ""
    
    # What this plugin provides
    capabilities: list[PluginCapability] = field(default_factory=list)
    
    # What this plugin requires
    dependencies: list[str] = field(default_factory=list)  # "plugin_name>=1.0.0"
    
    # Runtime requirements
    requires_runtime_version: str | None = None
    requires_llm_client_version: str | None = None
    
    # Entry point (for setuptools discovery)
    entry_point: str | None = None
    
    def provides_tools(self) -> bool:
        return any(c.type == PluginType.TOOLS for c in self.capabilities)
    
    def provides_operators(self) -> bool:
        return any(c.type == PluginType.OPERATOR for c in self.capabilities)
    
    def provides_policies(self) -> bool:
        return any(c.type == PluginType.POLICY for c in self.capabilities)
    
    def get_tool_names(self) -> list[str]:
        """Get all tool names provided by this plugin."""
        names = []
        for cap in self.capabilities:
            if cap.type == PluginType.TOOLS:
                names.extend(cap.tool_names)
        return names

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "capabilities": [
                {
                    "name": c.name,
                    "type": c.type.value,
                    "description": c.description,
                    "version": c.version,
                    "tool_names": c.tool_names,
                    "operator_id": c.operator_id,
                    "policy_names": c.policy_names,
                    "connector_names": c.connector_names,
                }
                for c in self.capabilities
            ],
            "dependencies": self.dependencies,
            "requires_runtime_version": self.requires_runtime_version,
            "requires_llm_client_version": self.requires_llm_client_version,
            "entry_point": self.entry_point,
        }


class Plugin(ABC):
    """Base class for plugins.
    
    Plugins extend the runtime with additional capabilities.
    They have lifecycle hooks for initialization and cleanup.
    """
    
    @property
    @abstractmethod
    def manifest(self) -> PluginManifest:
        """Return the plugin manifest."""
        ...
    
    async def on_load(self, registry: Any) -> None:
        """Called when the plugin is loaded.
        
        Use this to register tools, policies, etc. with the registry.
        """
        pass
    
    async def on_unload(self, registry: Any) -> None:
        """Called when the plugin is unloaded.
        
        Use this to clean up resources.
        """
        pass
    
    def get_tools(self) -> list[Tool]:
        """Return tools provided by this plugin.
        
        Override this to provide tools.
        """
        return []
    
    def get_policies(self) -> list[Any]:
        """Return policy rules provided by this plugin."""
        return []
    
    def get_operators(self) -> dict[str, Any]:
        """Return operators provided by this plugin.
        
        Returns:
            Dict mapping operator_id to operator implementation.
        """
        return {}


class ToolsPlugin(Plugin):
    """Convenience base class for plugins that only provide tools."""
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
    ):
        self._name = name
        self._version = version
        self._description = description
        self._tools: list[Tool] = []
    
    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            name=self._name,
            version=self._version,
            description=self._description,
            capabilities=[
                PluginCapability(
                    name=f"{self._name}_tools",
                    type=PluginType.TOOLS,
                    tool_names=[t.name for t in self._tools],
                )
            ],
        )
    
    def register_tool(self, tool: Tool) -> ToolsPlugin:
        """Register a tool with this plugin."""
        self._tools.append(tool)
        return self
    
    def get_tools(self) -> list[Tool]:
        return self._tools


__all__ = [
    "Plugin",
    "PluginManifest",
    "PluginType",
    "PluginCapability",
    "ToolsPlugin",
]

"""
Plugin registry for managing plugins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .types import Plugin, PluginManifest, PluginType, PluginCapability

if TYPE_CHECKING:
    from llm_client.tools.base import Tool


logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Information about a registered plugin."""
    manifest: PluginManifest
    plugin: Plugin
    loaded: bool = False
    enabled: bool = True


class PluginRegistry:
    """Registry for managing plugins.
    
    The registry:
    - Registers and tracks plugins
    - Manages plugin lifecycle (load/unload)
    - Provides access to plugin capabilities
    - Supports capability gating (allow/deny specific plugins)
    """
    
    def __init__(self):
        self._plugins: dict[str, PluginInfo] = {}  # name -> info
        self._allowed_plugins: set[str] | None = None  # None = all allowed
        self._denied_plugins: set[str] = set()
        self._tool_cache: dict[str, Tool] = {}  # tool_name -> tool
    
    def register(self, plugin: Plugin) -> PluginRegistry:
        """Register a plugin.
        
        Args:
            plugin: The plugin to register
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If a plugin with the same name is already registered
        """
        manifest = plugin.manifest
        
        if manifest.name in self._plugins:
            raise ValueError(f"Plugin '{manifest.name}' is already registered")
        
        self._plugins[manifest.name] = PluginInfo(
            manifest=manifest,
            plugin=plugin,
            loaded=False,
            enabled=True,
        )
        
        logger.info(f"Registered plugin: {manifest.name} v{manifest.version}")
        return self
    
    def unregister(self, name: str) -> bool:
        """Unregister a plugin.
        
        Returns:
            True if plugin was unregistered, False if not found
        """
        if name in self._plugins:
            info = self._plugins[name]
            
            # Remove tools from cache
            for tool_name in info.manifest.get_tool_names():
                self._tool_cache.pop(tool_name, None)
            
            del self._plugins[name]
            logger.info(f"Unregistered plugin: {name}")
            return True
        return False
    
    async def load(self, name: str) -> None:
        """Load a plugin (call on_load).
        
        This initializes the plugin and makes its capabilities available.
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found")
        
        info = self._plugins[name]
        if info.loaded:
            return
        
        # Call plugin's on_load
        await info.plugin.on_load(self)
        
        # Cache tools
        for tool in info.plugin.get_tools():
            self._tool_cache[tool.name] = tool
        
        info.loaded = True
        logger.info(f"Loaded plugin: {name}")
    
    async def unload(self, name: str) -> None:
        """Unload a plugin (call on_unload)."""
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not found")
        
        info = self._plugins[name]
        if not info.loaded:
            return
        
        # Call plugin's on_unload
        await info.plugin.on_unload(self)
        
        # Remove tools from cache
        for tool_name in info.manifest.get_tool_names():
            self._tool_cache.pop(tool_name, None)
        
        info.loaded = False
        logger.info(f"Unloaded plugin: {name}")
    
    async def load_all(self) -> None:
        """Load all registered plugins."""
        for name in self._plugins:
            if self.is_allowed(name):
                await self.load(name)
    
    async def unload_all(self) -> None:
        """Unload all plugins."""
        for name in list(self._plugins.keys()):
            await self.unload(name)
    
    def enable(self, name: str) -> None:
        """Enable a plugin."""
        if name in self._plugins:
            self._plugins[name].enabled = True
    
    def disable(self, name: str) -> None:
        """Disable a plugin."""
        if name in self._plugins:
            self._plugins[name].enabled = False
    
    def set_allowed_plugins(self, names: set[str] | None) -> None:
        """Set allowlist for plugins. None means all allowed."""
        self._allowed_plugins = names
    
    def set_denied_plugins(self, names: set[str]) -> None:
        """Set denylist for plugins."""
        self._denied_plugins = names
    
    def is_allowed(self, name: str) -> bool:
        """Check if a plugin is allowed."""
        if name in self._denied_plugins:
            return False
        if self._allowed_plugins is not None:
            return name in self._allowed_plugins
        return True
    
    def get(self, name: str) -> Plugin | None:
        """Get a plugin by name."""
        info = self._plugins.get(name)
        return info.plugin if info else None
    
    def get_manifest(self, name: str) -> PluginManifest | None:
        """Get a plugin's manifest."""
        info = self._plugins.get(name)
        return info.manifest if info else None
    
    def list_plugins(self) -> list[PluginManifest]:
        """List all registered plugins."""
        return [info.manifest for info in self._plugins.values()]
    
    def list_loaded(self) -> list[str]:
        """List names of loaded plugins."""
        return [name for name, info in self._plugins.items() if info.loaded]
    
    def get_tools(self) -> list[Tool]:
        """Get all tools from loaded plugins."""
        tools = []
        for info in self._plugins.values():
            if info.loaded and info.enabled and self.is_allowed(info.manifest.name):
                tools.extend(info.plugin.get_tools())
        return tools
    
    def get_tool(self, name: str) -> Tool | None:
        """Get a specific tool by name."""
        return self._tool_cache.get(name)
    
    def get_tool_names(self) -> list[str]:
        """Get names of all available tools."""
        return list(self._tool_cache.keys())
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is available and its plugin is allowed."""
        if tool_name not in self._tool_cache:
            return False
        
        # Find which plugin provides this tool
        for info in self._plugins.values():
            if tool_name in info.manifest.get_tool_names():
                return self.is_allowed(info.manifest.name) and info.enabled
        
        return True  # Tool exists but no plugin claims it (shouldn't happen)
    
    def get_policies(self) -> list[Any]:
        """Get all policies from loaded plugins."""
        policies = []
        for info in self._plugins.values():
            if info.loaded and info.enabled and self.is_allowed(info.manifest.name):
                policies.extend(info.plugin.get_policies())
        return policies
    
    def get_operators(self) -> dict[str, Any]:
        """Get all operators from loaded plugins."""
        operators = {}
        for info in self._plugins.values():
            if info.loaded and info.enabled and self.is_allowed(info.manifest.name):
                operators.update(info.plugin.get_operators())
        return operators
    
    def get_capabilities(
        self,
        plugin_type: PluginType | None = None,
    ) -> list[PluginCapability]:
        """Get all capabilities, optionally filtered by type."""
        capabilities = []
        for info in self._plugins.values():
            if not self.is_allowed(info.manifest.name):
                continue
            for cap in info.manifest.capabilities:
                if plugin_type is None or cap.type == plugin_type:
                    capabilities.append(cap)
        return capabilities


__all__ = [
    "PluginRegistry",
    "PluginInfo",
]

"""
Adapters for integrating external systems with Mainframe Orchestra.
"""

from .mcp_adapter import MCPOrchestra
from .composio_adapter import ComposioAdapter

__all__ = ["MCPOrchestra", "ComposioAdapter"]
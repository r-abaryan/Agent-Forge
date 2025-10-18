"""
AgentForge - Universal Multi-Agent Manager
Core modules for agent creation, management, and orchestration
"""

from .base_agent import BaseAgent
from .custom_agent import CustomAgent
from .agent_manager import AgentManager
from .agent_templates import AgentTemplates
from .agent_chain import AgentChain, WorkflowPresets
from .history_manager import HistoryManager
from .rag_integration import SimpleRAG

__all__ = [
    "BaseAgent",
    "CustomAgent",
    "AgentManager",
    "AgentTemplates",
    "AgentChain",
    "WorkflowPresets",
    "HistoryManager",
    "SimpleRAG",
]
__version__ = "2.0.0"


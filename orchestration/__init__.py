"""
AgentForge Orchestration - Workflow parsing and execution
Modules for parsing canvas workflows and executing them with AgentForge agents
"""

from .orchestration_parser import OrchestrationParser
from .workflow_executor import WorkflowExecutor
from .workflow_manager import WorkflowManager
from .workflow_templates import WorkflowTemplates

# Try to import workflow_code_generator if it exists
try:
    from .workflow_code_generator import WorkflowCodeGenerator
    __all__ = [
        "OrchestrationParser",
        "WorkflowExecutor",
        "WorkflowCodeGenerator",
        "WorkflowManager",
        "WorkflowTemplates",
    ]
except ImportError:
    __all__ = [
        "OrchestrationParser",
        "WorkflowExecutor",
        "WorkflowManager",
        "WorkflowTemplates",
    ]

__version__ = "1.0.0"


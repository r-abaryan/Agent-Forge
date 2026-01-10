"""
Workflow Analytics - Track and analyze workflow execution metrics
Clean, modular design with separation of concerns
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.logger_config import get_logger

logger = get_logger("agentforge.workflow_analytics")


@dataclass
class AgentExecutionMetrics:
    """Metrics for a single agent execution"""
    agent_name: str
    execution_time: float
    success: bool
    tokens_used: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class WorkflowExecutionMetrics:
    """Complete metrics for a workflow execution"""
    workflow_name: str
    execution_id: str
    start_time: str
    end_time: Optional[str] = None
    total_duration: Optional[float] = None
    success: bool = False
    agent_metrics: List[Dict[str, Any]] = None
    total_tokens: Optional[int] = None
    total_agents: int = 0
    successful_agents: int = 0
    failed_agents: int = 0
    pass_mode: str = "cumulative"
    input_length: Optional[int] = None
    output_length: Optional[int] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.agent_metrics is None:
            self.agent_metrics = []
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
        if self.execution_id is None:
            self.execution_id = f"{int(time.time() * 1000)}"


class WorkflowAnalytics:
    """
    Track and analyze workflow execution metrics.
    Handles storage, retrieval, and aggregation of analytics data.
    """
    
    def __init__(self, storage_dir: str = "workflow_analytics"):
        """
        Initialize analytics system.
        
        Args:
            storage_dir: Directory to store analytics data
        """
        self.storage_dir = Path(storage_dir)
        self._ensure_storage_dir()
        logger.info(f"Workflow analytics initialized: {storage_dir}")
    
    def _ensure_storage_dir(self):
        """Create storage directory if it doesn't exist"""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            gitignore_path = self.storage_dir / ".gitignore"
            if not gitignore_path.exists():
                gitignore_path.write_text("*\n!.gitignore\n")
        except Exception as e:
            logger.warning(f"Could not create analytics directory: {e}")
    
    def start_execution(
        self,
        workflow_name: str,
        pass_mode: str = "cumulative",
        input_length: Optional[int] = None
    ) -> WorkflowExecutionMetrics:
        """
        Start tracking a workflow execution.
        
        Args:
            workflow_name: Name of the workflow
            pass_mode: Data passing mode
            input_length: Length of input text
        
        Returns:
            WorkflowExecutionMetrics instance
        """
        metrics = WorkflowExecutionMetrics(
            workflow_name=workflow_name,
            execution_id=f"{int(time.time() * 1000)}",
            pass_mode=pass_mode,
            input_length=input_length
        )
        logger.debug(f"Started tracking execution: {metrics.execution_id}")
        return metrics
    
    def record_agent_execution(
        self,
        metrics: WorkflowExecutionMetrics,
        agent_name: str,
        execution_time: float,
        success: bool,
        tokens_used: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """
        Record metrics for a single agent execution.
        
        Args:
            metrics: WorkflowExecutionMetrics instance
            agent_name: Name of the agent
            execution_time: Execution time in seconds
            success: Whether execution succeeded
            tokens_used: Number of tokens used
            error_message: Error message if failed
        """
        agent_metrics = AgentExecutionMetrics(
            agent_name=agent_name,
            execution_time=execution_time,
            success=success,
            tokens_used=tokens_used,
            error_message=error_message
        )
        
        metrics.agent_metrics.append(asdict(agent_metrics))
        metrics.total_agents += 1
        
        if success:
            metrics.successful_agents += 1
        else:
            metrics.failed_agents += 1
        
        if tokens_used:
            metrics.total_tokens = (metrics.total_tokens or 0) + tokens_used
        
        logger.debug(f"Recorded agent execution: {agent_name} ({execution_time:.2f}s)")
    
    def finish_execution(
        self,
        metrics: WorkflowExecutionMetrics,
        success: bool,
        output_length: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """
        Finish tracking a workflow execution.
        
        Args:
            metrics: WorkflowExecutionMetrics instance
            success: Whether workflow succeeded
            output_length: Length of output text
            error_message: Error message if failed
        """
        metrics.end_time = datetime.now().isoformat()
        metrics.success = success
        metrics.output_length = output_length
        metrics.error_message = error_message
        
        if metrics.start_time:
            start = datetime.fromisoformat(metrics.start_time)
            end = datetime.fromisoformat(metrics.end_time)
            metrics.total_duration = (end - start).total_seconds()
        
        self._save_metrics(metrics)
        logger.info(
            f"Finished execution {metrics.execution_id}: "
            f"{'SUCCESS' if success else 'FAILED'} "
            f"({metrics.total_duration:.2f}s, {metrics.total_agents} agents)"
        )
    
    def _save_metrics(self, metrics: WorkflowExecutionMetrics):
        """
        Save execution metrics to disk.
        
        Args:
            metrics: WorkflowExecutionMetrics instance
        """
        try:
            safe_name = self._sanitize_filename(metrics.workflow_name)
            file_path = self.storage_dir / f"{safe_name}_{metrics.execution_id}.json"
            
            metrics_dict = asdict(metrics)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize workflow name for filename"""
        safe = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        return safe.strip().replace(' ', '_')[:100]
    
    def get_workflow_analytics(
        self,
        workflow_name: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get analytics for a specific workflow.
        
        Args:
            workflow_name: Name of the workflow
            limit: Maximum number of executions to return
        
        Returns:
            List of execution metrics
        """
        try:
            safe_name = self._sanitize_filename(workflow_name)
            pattern = f"{safe_name}_*.json"
            
            executions = []
            for file_path in self.storage_dir.glob(pattern):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    executions.append(metrics)
                except Exception as e:
                    logger.warning(f"Error reading metrics file {file_path}: {e}")
            
            executions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
            return executions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting workflow analytics: {e}")
            return []
    
    def get_all_analytics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get analytics for all workflows.
        
        Args:
            limit: Maximum number of executions to return
        
        Returns:
            List of execution metrics
        """
        try:
            executions = []
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    executions.append(metrics)
                except Exception as e:
                    logger.warning(f"Error reading metrics file {file_path}: {e}")
            
            executions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
            return executions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting all analytics: {e}")
            return []
    
    def get_workflow_summary(self, workflow_name: str) -> Dict[str, Any]:
        """
        Get aggregated summary statistics for a workflow.
        
        Args:
            workflow_name: Name of the workflow
        
        Returns:
            Dictionary with summary statistics
        """
        executions = self.get_workflow_analytics(workflow_name, limit=1000)
        
        if not executions:
            return {
                "workflow_name": workflow_name,
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "total_tokens": 0,
                "avg_agents": 0.0
            }
        
        successful = sum(1 for e in executions if e.get('success', False))
        durations = [e.get('total_duration', 0) for e in executions if e.get('total_duration')]
        tokens = [e.get('total_tokens', 0) for e in executions if e.get('total_tokens')]
        agent_counts = [e.get('total_agents', 0) for e in executions]
        
        return {
            "workflow_name": workflow_name,
            "total_executions": len(executions),
            "successful_executions": successful,
            "failed_executions": len(executions) - successful,
            "success_rate": (successful / len(executions) * 100) if executions else 0.0,
            "avg_duration": sum(durations) / len(durations) if durations else 0.0,
            "min_duration": min(durations) if durations else 0.0,
            "max_duration": max(durations) if durations else 0.0,
            "total_tokens": sum(tokens),
            "avg_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
            "avg_agents": sum(agent_counts) / len(agent_counts) if agent_counts else 0.0,
            "last_execution": executions[0].get('start_time') if executions else None
        }
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """
        Get overall analytics summary across all workflows.
        
        Returns:
            Dictionary with overall statistics
        """
        all_executions = self.get_all_analytics(limit=10000)
        
        if not all_executions:
            return {
                "total_executions": 0,
                "total_workflows": 0,
                "overall_success_rate": 0.0,
                "avg_duration": 0.0,
                "total_tokens": 0
            }
        
        workflow_names = set(e.get('workflow_name') for e in all_executions)
        successful = sum(1 for e in all_executions if e.get('success', False))
        durations = [e.get('total_duration', 0) for e in all_executions if e.get('total_duration')]
        tokens = [e.get('total_tokens', 0) for e in all_executions if e.get('total_tokens')]
        
        return {
            "total_executions": len(all_executions),
            "total_workflows": len(workflow_names),
            "successful_executions": successful,
            "failed_executions": len(all_executions) - successful,
            "overall_success_rate": (successful / len(all_executions) * 100) if all_executions else 0.0,
            "avg_duration": sum(durations) / len(durations) if durations else 0.0,
            "total_tokens": sum(tokens),
            "avg_tokens_per_execution": sum(tokens) / len(tokens) if tokens else 0.0
        }
    
    def delete_workflow_analytics(self, workflow_name: str) -> bool:
        """
        Delete all analytics for a workflow.
        
        Args:
            workflow_name: Name of the workflow
        
        Returns:
            True if successful, False otherwise
        """
        try:
            safe_name = self._sanitize_filename(workflow_name)
            pattern = f"{safe_name}_*.json"
            
            deleted = 0
            for file_path in self.storage_dir.glob(pattern):
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Error deleting {file_path}: {e}")
            
            logger.info(f"Deleted {deleted} analytics files for workflow '{workflow_name}'")
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Error deleting workflow analytics: {e}")
            return False

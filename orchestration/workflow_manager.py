"""
Workflow Manager - Save, load, and manage canvas workflows
Handles persistent storage with error handling
"""

import json
import os
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for logger import
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.logger_config import get_logger

logger = get_logger("agentforge.workflow_manager")


class WorkflowManager:
    """
    Manages canvas workflow lifecycle: save, load, list, delete, import/export.
    Uses JSON file storage with atomic writes.
    """
    
    def __init__(self, storage_dir: str = "workflows"):
        """
        Initialize workflow manager.
        
        Args:
            storage_dir: Directory to store workflow JSON files
        """
        self.storage_dir = Path(storage_dir)
        self._ensure_storage_dir()
        self._workflow_cache: Dict[str, Dict] = {}
    
    def _ensure_storage_dir(self):
        """Create storage directory if it doesn't exist"""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
            gitignore_path = self.storage_dir / ".gitignore"
            if not gitignore_path.exists():
                gitignore_path.write_text("*\n!.gitignore\n")
            
        except Exception as e:
            logger.warning(f"Could not create storage directory: {e}")
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize workflow name for filename"""
        safe = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        return safe.strip().replace(' ', '_')[:100]
    
    def save_workflow(
        self,
        workflow_json: Dict[str, Any],
        workflow_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        Save a workflow to disk.
        
        Args:
            workflow_json: Canvas workflow JSON
            workflow_name: Optional name override
            description: Optional description
        
        Returns:
            True if successful, False otherwise
        """
        try:
            name = workflow_name or workflow_json.get("name", "Untitled Workflow")
            safe_name = self._sanitize_filename(name)
            file_path = self.storage_dir / f"{safe_name}.json"
            
            workflow_data = {
                "name": name,
                "description": description or workflow_json.get("notes", ""),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": workflow_json.get("version", "1.0.0"),
                "format": workflow_json.get("format", "website-playground"),
                "workflow": workflow_json,
                "metadata": {
                    "agent_count": len(workflow_json.get("graph", {}).get("cells", [])),
                    "saved_by": "workflow_manager"
                }
            }
            
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, indent=2, ensure_ascii=False)
            
            temp_path.replace(file_path)
            self._workflow_cache[safe_name] = workflow_data
            
            logger.info(f"Workflow '{name}' saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving workflow '{name}': {e}")
            return False
    
    def load_workflow(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a workflow from disk.
        
        Args:
            workflow_name: Name of workflow to load
        
        Returns:
            Workflow JSON or None if not found
        """
        try:
            safe_name = self._sanitize_filename(workflow_name)
            
            if safe_name in self._workflow_cache:
                return self._workflow_cache[safe_name].get("workflow")
            
            file_path = self.storage_dir / f"{safe_name}.json"
            if not file_path.exists():
                logger.warning(f"Workflow '{workflow_name}' not found")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            self._workflow_cache[safe_name] = workflow_data
            return workflow_data.get("workflow")
            
        except Exception as e:
            logger.error(f"Error loading workflow '{workflow_name}': {e}")
            return None
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all saved workflows.
        
        Returns:
            List of workflow metadata
        """
        workflows = []
        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        workflow_data = json.load(f)
                    
                    workflows.append({
                        "name": workflow_data.get("name", file_path.stem),
                        "description": workflow_data.get("description", ""),
                        "created_at": workflow_data.get("created_at", ""),
                        "updated_at": workflow_data.get("updated_at", ""),
                        "version": workflow_data.get("version", "1.0.0"),
                        "agent_count": workflow_data.get("metadata", {}).get("agent_count", 0)
                    })
                except Exception as e:
                    logger.warning(f"Error reading workflow file {file_path}: {e}")
            
            workflows.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return workflows
            
        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            return []
    
    def workflow_exists(self, workflow_name: str) -> bool:
        """Check if workflow exists"""
        safe_name = self._sanitize_filename(workflow_name)
        file_path = self.storage_dir / f"{safe_name}.json"
        return file_path.exists()
    
    def delete_workflow(self, workflow_name: str) -> bool:
        """
        Delete a workflow.
        
        Args:
            workflow_name: Name of workflow to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            safe_name = self._sanitize_filename(workflow_name)
            file_path = self.storage_dir / f"{safe_name}.json"
            
            if not file_path.exists():
                logger.warning(f"Workflow '{workflow_name}' not found")
                return False
            
            file_path.unlink()
            self._workflow_cache.pop(safe_name, None)
            
            logger.info(f"Workflow '{workflow_name}' deleted")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting workflow '{workflow_name}': {e}")
            return False
    
    def export_workflow(self, workflow_name: str, export_path: str) -> bool:
        """
        Export workflow to JSON file.
        
        Args:
            workflow_name: Name of workflow to export
            export_path: Path to save exported file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            workflow_json = self.load_workflow(workflow_name)
            if not workflow_json:
                return False
            
            export_file = Path(export_path)
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_json, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Workflow '{workflow_name}' exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting workflow '{workflow_name}': {e}")
            return False
    
    def import_workflow(self, import_path: str, workflow_name: Optional[str] = None) -> bool:
        """
        Import workflow from JSON file.
        
        Args:
            import_path: Path to workflow JSON file
            workflow_name: Optional name override
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import_file = Path(import_path)
            if not import_file.exists():
                logger.error(f"Import file not found: {import_path}")
                return False
            
            with open(import_file, 'r', encoding='utf-8') as f:
                workflow_json = json.load(f)
            
            return self.save_workflow(workflow_json, workflow_name=workflow_name)
            
        except Exception as e:
            logger.error(f"Error importing workflow from {import_path}: {e}")
            return False
    
    def export_all_workflows(self, export_path: str) -> bool:
        """
        Export all workflows to a ZIP file.
        
        Args:
            export_path: Path to save ZIP file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            workflows = self.list_workflows()
            if not workflows:
                logger.warning("No workflows to export")
                return False
            
            export_file = Path(export_path)
            with zipfile.ZipFile(export_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for workflow in workflows:
                    workflow_json = self.load_workflow(workflow["name"])
                    if workflow_json:
                        safe_name = self._sanitize_filename(workflow["name"])
                        zipf.writestr(
                            f"{safe_name}.json",
                            json.dumps(workflow_json, indent=2, ensure_ascii=False)
                        )
            
            logger.info(f"Exported {len(workflows)} workflows to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting workflows: {e}")
            return False
    
    def get_workflow_metadata(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get workflow metadata without loading full workflow"""
        safe_name = self._sanitize_filename(workflow_name)
        
        if safe_name in self._workflow_cache:
            data = self._workflow_cache[safe_name]
            return {
                "name": data.get("name"),
                "description": data.get("description"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "version": data.get("version"),
                "agent_count": data.get("metadata", {}).get("agent_count", 0)
            }
        
        file_path = self.storage_dir / f"{safe_name}.json"
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                "name": data.get("name"),
                "description": data.get("description"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "version": data.get("version"),
                "agent_count": data.get("metadata", {}).get("agent_count", 0)
            }
        except Exception as e:
            logger.error(f"Error reading workflow metadata: {e}")
            return None


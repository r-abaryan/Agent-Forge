"""
Agent Manager - Save, load, and manage custom agents
Handles persistent storage with error handling
"""

import json
import os
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from .custom_agent import CustomAgent
from .logger_config import get_logger

logger = get_logger("agentforge.agent_manager")


class AgentManager:
    """
    Manages custom agent lifecycle: create, save, load, list, delete.
    Uses JSON file storage with atomic writes.
    """
    
    def __init__(self, storage_dir: str = "custom_agents"):
        """
        Initialize agent manager.
        
        Args:
            storage_dir: Directory to store agent JSON files
        """
        self.storage_dir = Path(storage_dir)
        self._ensure_storage_dir()
        # Cache for agent data to reduce file I/O
        self._agent_data_cache: Dict[str, Dict] = {}
    
    def _ensure_storage_dir(self):
        """Create storage directory if it doesn't exist"""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Create .gitignore to prevent committing user data
            gitignore_path = self.storage_dir / ".gitignore"
            if not gitignore_path.exists():
                gitignore_path.write_text("# Ignore all custom agent files\n*.json\n")
        except Exception as e:
            logger.warning(f"Could not create storage directory: {e}")
    
    def save_agent(self, agent: CustomAgent) -> bool:
        """
        Save a custom agent to disk.
        
        Args:
            agent: CustomAgent instance
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Sanitize filename
            safe_name = self._sanitize_filename(agent.name)
            file_path = self.storage_dir / f"{safe_name}.json"
            
            # Export agent data
            agent_data = agent.to_dict()
            
            # Write atomically (write to temp, then rename)
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(agent_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_path.replace(file_path)
            
            # Update cache
            self._agent_data_cache[safe_name] = agent_data
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving agent '{agent.name}': {e}")
            return False
    
    def load_agent(self, agent_name: str, llm=None) -> Optional[CustomAgent]:
        """
        Load a custom agent from disk (with caching).
        
        Args:
            agent_name: Name of the agent to load
            llm: LLM instance to attach to the agent
        
        Returns:
            CustomAgent instance or None if not found/invalid
        """
        try:
            safe_name = self._sanitize_filename(agent_name)
            file_path = self.storage_dir / f"{safe_name}.json"
            
            if not file_path.exists():
                logger.warning(f"Agent '{agent_name}' not found")
                return None
            
            # Check cache first
            if safe_name in self._agent_data_cache:
                agent_data = self._agent_data_cache[safe_name]
            else:
                # Load from disk and cache
                with open(file_path, 'r', encoding='utf-8') as f:
                    agent_data = json.load(f)
                self._agent_data_cache[safe_name] = agent_data
            
            # Create agent from data (LLM attached at creation time)
            agent = CustomAgent.from_dict(agent_data, llm=llm)
            return agent
            
        except Exception as e:
            logger.error(f"Error loading agent '{agent_name}': {e}")
            return None
    
    def list_agents(self) -> List[Dict[str, str]]:
        """
        List all saved custom agents.
        
        Returns:
            List of dictionaries with agent metadata (name, role)
        """
        agents = []
        
        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    safe_name = file_path.stem
                    if safe_name in self._agent_data_cache:
                        agent_data = self._agent_data_cache[safe_name]
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            agent_data = json.load(f)
                        self._agent_data_cache[safe_name] = agent_data

                    agents.append({
                        "name": agent_data.get("name", "Unknown"),
                        "role": agent_data.get("role", "No role specified"),
                        "filename": safe_name
                    })
                except Exception as e:
                    logger.error(f"Error reading {file_path.name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
        
        return sorted(agents, key=lambda x: x["name"].lower())
    
    def delete_agent(self, agent_name: str) -> bool:
        """
        Delete a custom agent.
        
        Args:
            agent_name: Name of the agent to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            safe_name = self._sanitize_filename(agent_name)
            file_path = self.storage_dir / f"{safe_name}.json"
            
            if not file_path.exists():
                logger.warning(f"Agent '{agent_name}' not found")
                return False
            
            file_path.unlink()
            # Invalidate cache
            if safe_name in self._agent_data_cache:
                del self._agent_data_cache[safe_name]
            return True
            
        except Exception as e:
            logger.error(f"Error deleting agent '{agent_name}': {e}")
            return False
    
    def agent_exists(self, agent_name: str) -> bool:
        """Check if an agent exists"""
        safe_name = self._sanitize_filename(agent_name)
        file_path = self.storage_dir / f"{safe_name}.json"
        return file_path.exists()
    
    def update_agent(self, old_name: str, agent: CustomAgent) -> bool:
        """
        Update an existing agent (supports renaming).
        
        Args:
            old_name: Original agent name
            agent: Updated CustomAgent instance
        
        Returns:
            True if successful, False otherwise
        """
        try:
            old_safe_name = self._sanitize_filename(old_name)
            # If name changed, delete old file and invalidate cache
            if old_name != agent.name:
                self.delete_agent(old_name)
            
            # Save updated agent (cache will be updated in save_agent)
            return self.save_agent(agent)
            
        except Exception as e:
            logger.error(f"Error updating agent '{old_name}': {e}")
            return False
    
    def get_agent_data(self, agent_name: str) -> Optional[Dict[str, str]]:
        """
        Get raw agent data without instantiating the agent.
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            Dictionary with agent data or None if not found
        """
        try:
            safe_name = self._sanitize_filename(agent_name)
            file_path = self.storage_dir / f"{safe_name}.json"
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error getting agent data '{agent_name}': {e}")
            return None
    
    def export_agent(self, agent_name: str, output_path: str) -> bool:
        """
        Export an agent to a JSON file.
        
        Args:
            agent_name: Name of the agent to export
            output_path: Path to save the exported agent
        
        Returns:
            True if successful, False otherwise
        """
        try:
            agent_data = self.get_agent_data(agent_name)
            if not agent_data:
                return False
            
            # Add metadata
            export_data = {
                "agentforge_version": "1.0",
                "export_date": datetime.now().isoformat(),
                "agent": agent_data
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting agent '{agent_name}': {e}")
            return False
    
    def import_agent(self, import_path: str, llm=None, overwrite: bool = False) -> bool:
        """
        Import an agent from a JSON file.
        
        Args:
            import_path: Path to the agent file
            llm: LLM instance to attach
            overwrite: Whether to overwrite existing agent
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Extract agent data
            if "agent" in import_data:
                agent_data = import_data["agent"]
            else:
                # Legacy format
                agent_data = import_data
            
            # Check if agent exists
            agent_name = agent_data.get("name", "")
            if self.agent_exists(agent_name) and not overwrite:
                logger.warning(f"Agent '{agent_name}' already exists. Use overwrite=True to replace.")
                return False
            
            # Create and save agent
            agent = CustomAgent.from_dict(agent_data, llm=llm)
            return self.save_agent(agent)
            
        except Exception as e:
            logger.error(f"Error importing agent: {e}")
            return False
    
    def export_all_agents(self, output_path: str = "agents_backup.zip") -> bool:
        """
        Export all agents to a zip file.
        
        Args:
            output_path: Path to save the zip file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add metadata
                metadata = {
                    "agentforge_version": "1.0",
                    "export_date": datetime.now().isoformat(),
                    "agent_count": len(self.list_agents())
                }
                zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
                
                # Add all agent files
                for file_path in self.storage_dir.glob("*.json"):
                    zipf.write(file_path, file_path.name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting all agents: {e}")
            return False
    
    def import_all_agents(self, import_path: str, llm=None, overwrite: bool = False) -> Dict[str, any]:
        """
        Import agents from a zip file.
        
        Args:
            import_path: Path to the zip file
            llm: LLM instance to attach
            overwrite: Whether to overwrite existing agents
        
        Returns:
            Dictionary with import results
        """
        results = {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "agents": []
        }
        
        try:
            with zipfile.ZipFile(import_path, 'r') as zipf:
                # Get all JSON files (except metadata)
                agent_files = [f for f in zipf.namelist() if f.endswith('.json') and f != 'metadata.json']
                
                for filename in agent_files:
                    try:
                        # Read agent data
                        with zipf.open(filename) as f:
                            agent_data = json.load(f)
                        
                        agent_name = agent_data.get("name", "")
                        
                        # Check if exists
                        if self.agent_exists(agent_name) and not overwrite:
                            results["skipped"] += 1
                            results["agents"].append({
                                "name": agent_name,
                                "status": "skipped",
                                "reason": "Already exists"
                            })
                            continue
                        
                        # Import agent
                        agent = CustomAgent.from_dict(agent_data, llm=llm)
                        if self.save_agent(agent):
                            results["success"] += 1
                            results["agents"].append({
                                "name": agent_name,
                                "status": "success"
                            })
                        else:
                            results["failed"] += 1
                            results["agents"].append({
                                "name": agent_name,
                                "status": "failed",
                                "reason": "Save failed"
                            })
                            
                    except Exception as e:
                        results["failed"] += 1
                        results["agents"].append({
                            "name": filename,
                            "status": "failed",
                            "reason": str(e)
                        })
        
        except Exception as e:
            logger.error(f"Error importing agents: {e}")
            results["error"] = str(e)
        
        return results
    
    def clear_cache(self):
        """Clear the agent data cache (useful for memory management or forced refresh)"""
        self._agent_data_cache.clear()
    
    def _sanitize_filename(self, name: str) -> str:
        """
        Convert agent name to safe filename.
        Removes special characters and limits length.
        """
        # Replace spaces with underscores, remove special chars
        safe_name = name.lower().strip()
        safe_name = safe_name.replace(' ', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
        
        # Limit length
        return safe_name[:50]


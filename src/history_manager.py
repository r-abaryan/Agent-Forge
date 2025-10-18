"""
History Manager - Track and save conversation history
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class HistoryManager:
    """Manages conversation history and logging"""
    
    def __init__(self, history_dir: str = "history"):
        """
        Initialize history manager.
        
        Args:
            history_dir: Directory to store history files
        """
        self.history_dir = Path(history_dir)
        self._ensure_history_dir()
    
    def _ensure_history_dir(self):
        """Create history directory if it doesn't exist"""
        try:
            self.history_dir.mkdir(parents=True, exist_ok=True)
            
            # Create .gitignore
            gitignore_path = self.history_dir / ".gitignore"
            if not gitignore_path.exists():
                gitignore_path.write_text("# Ignore all history files\n*.json\n")
        except Exception as e:
            print(f"Warning: Could not create history directory: {e}")
    
    def save_conversation(
        self, 
        input_text: str, 
        context: str, 
        agents_used: List[str],
        responses: Dict[str, str]
    ) -> bool:
        """
        Save a conversation to history.
        
        Args:
            input_text: User input
            context: Context provided
            agents_used: List of agent names used
            responses: Dictionary mapping agent names to responses
        
        Returns:
            True if successful, False otherwise
        """
        try:
            timestamp = datetime.now()
            
            conversation = {
                "timestamp": timestamp.isoformat(),
                "input": input_text,
                "context": context,
                "agents_used": agents_used,
                "responses": responses,
                "metadata": {
                    "agent_count": len(agents_used),
                    "date": timestamp.strftime("%Y-%m-%d"),
                    "time": timestamp.strftime("%H:%M:%S")
                }
            }
            
            # Create filename with timestamp
            filename = f"conversation_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            file_path = self.history_dir / filename
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def list_conversations(self, limit: int = 50) -> List[Dict[str, any]]:
        """
        List recent conversations.
        
        Args:
            limit: Maximum number of conversations to return
        
        Returns:
            List of conversation summaries
        """
        conversations = []
        
        try:
            # Get all conversation files
            files = sorted(
                self.history_dir.glob("conversation_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Load summaries
            for file_path in files[:limit]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Create summary
                    conversations.append({
                        "filename": file_path.name,
                        "timestamp": data.get("timestamp", "Unknown"),
                        "date": data.get("metadata", {}).get("date", "Unknown"),
                        "time": data.get("metadata", {}).get("time", "Unknown"),
                        "input_preview": data.get("input", "")[:100] + "..." if len(data.get("input", "")) > 100 else data.get("input", ""),
                        "agents_used": data.get("agents_used", []),
                        "agent_count": len(data.get("agents_used", []))
                    })
                    
                except Exception as e:
                    print(f"Error reading {file_path.name}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error listing conversations: {e}")
        
        return conversations
    
    def get_conversation(self, filename: str) -> Optional[Dict[str, any]]:
        """
        Get full conversation details.
        
        Args:
            filename: Conversation filename
        
        Returns:
            Full conversation data or None if not found
        """
        try:
            file_path = self.history_dir / filename
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error getting conversation '{filename}': {e}")
            return None
    
    def delete_conversation(self, filename: str) -> bool:
        """
        Delete a conversation from history.
        
        Args:
            filename: Conversation filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.history_dir / filename
            
            if not file_path.exists():
                return False
            
            file_path.unlink()
            return True
            
        except Exception as e:
            print(f"Error deleting conversation '{filename}': {e}")
            return False
    
    def clear_all_history(self) -> bool:
        """
        Clear all conversation history.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for file_path in self.history_dir.glob("conversation_*.json"):
                file_path.unlink()
            return True
            
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False
    
    def export_history(self, output_file: str = "history_export.json") -> bool:
        """
        Export all history to a single file.
        
        Args:
            output_file: Output filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conversations = []
            
            for file_path in sorted(self.history_dir.glob("conversation_*.json")):
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversations.append(json.load(f))
            
            export_data = {
                "export_date": datetime.now().isoformat(),
                "total_conversations": len(conversations),
                "conversations": conversations
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error exporting history: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            files = list(self.history_dir.glob("conversation_*.json"))
            total_conversations = len(files)
            
            agent_usage = {}
            total_agents_used = 0
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    agents = data.get("agents_used", [])
                    total_agents_used += len(agents)
                    
                    for agent in agents:
                        agent_usage[agent] = agent_usage.get(agent, 0) + 1
                        
                except Exception:
                    continue
            
            # Sort agents by usage
            most_used_agents = sorted(
                agent_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                "total_conversations": total_conversations,
                "total_agent_invocations": total_agents_used,
                "average_agents_per_conversation": round(total_agents_used / total_conversations, 2) if total_conversations > 0 else 0,
                "unique_agents_used": len(agent_usage),
                "most_used_agents": [{"agent": name, "count": count} for name, count in most_used_agents]
            }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                "total_conversations": 0,
                "total_agent_invocations": 0,
                "average_agents_per_conversation": 0,
                "unique_agents_used": 0,
                "most_used_agents": []
            }


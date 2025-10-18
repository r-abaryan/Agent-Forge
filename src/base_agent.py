"""
Base Agent Class for AgentForge
General-purpose agent system for any domain
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, role: str, llm=None):
        self.name = name
        self.role = role
        self.llm = llm
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the specialized system prompt for this agent"""
        pass
    
    @abstractmethod
    def process(self, input_text: str, context: str = "") -> Dict[str, Any]:
        """
        Process input and return response
        
        Args:
            input_text: Main input text
            context: Additional context
        
        Returns:
            Dictionary with response and metadata
        """
        pass
    
    def format_prompt(self, input_text: str, context: str = "") -> str:
        """Format the complete prompt for the LLM"""
        system = self.get_system_prompt()
        
        user_input = f"""Input: {input_text}

Context: {context if context else "None provided"}

Provide your response:"""
        
        return f"{system}\n\n{user_input}"
    
    def __str__(self):
        return f"{self.name} ({self.role})"


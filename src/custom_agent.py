"""
Custom Agent - User-defined agents with custom prompts
Adapted from CyberXP for general-purpose use
"""

from typing import Dict, Any
from .base_agent import BaseAgent
import re


class CustomAgent(BaseAgent):
    """
    Custom agent that users can create with their own prompts.
    Includes safety checks and flexible configuration.
    """
    
    MAX_PROMPT_LENGTH = 4000
    MAX_NAME_LENGTH = 100
    MAX_ROLE_LENGTH = 200
    
    def __init__(self, name: str, role: str, system_prompt: str, llm=None, few_shot_examples: str = ""):
        """
        Initialize a custom agent with validation.
        
        Args:
            name: Agent name (max 100 chars)
            role: Agent role/specialty (max 200 chars)
            system_prompt: Custom system prompt (max 4000 chars)
            llm: Language model instance
            few_shot_examples: Optional few-shot examples
        
        Raises:
            ValueError: If inputs fail validation
        """
        # Validate and sanitize inputs
        name = self._validate_name(name)
        role = self._validate_role(role)
        system_prompt = self._validate_prompt(system_prompt)
        few_shot_examples = self._validate_prompt(few_shot_examples) if few_shot_examples else ""
        
        super().__init__(name, role, llm)
        self._system_prompt = system_prompt
        self._few_shot_examples = few_shot_examples
    
    def _validate_name(self, name: str) -> str:
        """Validate and sanitize agent name"""
        if not name or not isinstance(name, str):
            raise ValueError("Agent name must be a non-empty string")
        
        name = name.strip()
        
        if len(name) > self.MAX_NAME_LENGTH:
            raise ValueError(f"Agent name must be <= {self.MAX_NAME_LENGTH} characters")
        
        # Only allow alphanumeric, spaces, hyphens, underscores
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', name):
            raise ValueError("Agent name can only contain letters, numbers, spaces, hyphens, and underscores")
        
        return name
    
    def _validate_role(self, role: str) -> str:
        """Validate and sanitize role description"""
        if not role or not isinstance(role, str):
            raise ValueError("Agent role must be a non-empty string")
        
        role = role.strip()
        
        if len(role) > self.MAX_ROLE_LENGTH:
            raise ValueError(f"Agent role must be <= {self.MAX_ROLE_LENGTH} characters")
        
        return role
    
    def _validate_prompt(self, prompt: str) -> str:
        """Validate and sanitize system prompt"""
        if not prompt or not isinstance(prompt, str):
            raise ValueError("System prompt must be a non-empty string")
        
        prompt = prompt.strip()
        
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            raise ValueError(f"System prompt must be <= {self.MAX_PROMPT_LENGTH} characters")
        
        # Check for potentially harmful patterns (basic safety)
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise ValueError("System prompt contains potentially unsafe patterns")
        
        return prompt
    
    def get_system_prompt(self) -> str:
        """Return the custom system prompt"""
        base_prompt = f"""You are {self.name}, a specialized agent.
Your role: {self.role}

{self._system_prompt}"""
        
        if self._few_shot_examples:
            base_prompt += f"\n\nExamples:\n{self._few_shot_examples}"
        
        return base_prompt
    
    def process(self, input_text: str, context: str = "") -> Dict[str, Any]:
        """
        Process input using the custom agent's prompt.
        
        Args:
            input_text: Input text to process
            context: Additional context
        
        Returns:
            Dictionary with response and metadata
        """
        if not self.llm:
            return {
                "agent": self.name,
                "role": self.role,
                "response": "Error: No LLM configured for this agent",
                "success": False
            }
        
        try:
            # Build robust prompt with clear structure
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            # Create well-structured system prompt
            # Keep it simple and direct to avoid breaking model output
            system_prompt = self._system_prompt.strip()
            
            # Build clear user message with length limits
            # Truncate context if too long to avoid overwhelming the model
            max_context_length = 2000
            truncated_context = context.strip()[:max_context_length] if len(context.strip()) > max_context_length else context.strip()
            
            if truncated_context:
                user_message = f"{input_text.strip()}\n\nContext: {truncated_context}"
            else:
                user_message = input_text.strip()
            
            # Limit user message length
            max_user_message = 3000
            if len(user_message) > max_user_message:
                user_message = user_message[:max_user_message] + "... [truncated]"
            
            # Use simple, reliable prompt template
            # Avoid complex formatting that might confuse the model
            # CRITICAL: Do NOT include "Response Guidelines" in output - agents should NOT echo this
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\n**IMPORTANT**: Your response should ONLY contain the actual answer/data. Do NOT include any guidelines, instructions, or meta-commentary in your response. Be direct and factual."),
                ("human", user_message)
            ])
            
            # Create chain with output parser
            chain = prompt | self.llm | StrOutputParser()
            
            # Invoke chain with empty dict (all variables in template)
            response = chain.invoke({})
            
            # Clean response: remove common artifacts
            response_clean = response.strip()
            
            # Remove any accidental prompt echoes at start
            if response_clean.startswith(("System:", "Human:", "Assistant:")):
                lines = response_clean.split('\n', 1)
                if len(lines) > 1:
                    response_clean = lines[1].strip()
            
            # Remove repetitive patterns FIRST (before length check)
            lines = response_clean.split('\n')
            seen_lines = set()
            filtered_lines = []
            consecutive_duplicates = 0
            
            for i, line in enumerate(lines):
                line_stripped = line.strip().lower()
                # Check for exact duplicates or very similar lines
                if line_stripped:
                    # Check if this line is too similar to recent lines
                    is_duplicate = False
                    for seen in list(seen_lines)[-5:]:  # Check last 5 seen lines
                        if len(line_stripped) > 30 and seen and len(seen) > 30:
                            # Simple similarity check
                            words_current = set(line_stripped.split()[:10])
                            words_seen = set(seen.split()[:10])
                            if len(words_current & words_seen) > 5:  # More than 5 common words
                                is_duplicate = True
                                consecutive_duplicates += 1
                                break
                    
                    if is_duplicate and consecutive_duplicates > 1:
                        continue  # Skip this duplicate line
                    elif is_duplicate:
                        consecutive_duplicates += 1
                    else:
                        consecutive_duplicates = 0
                        if len(line_stripped) > 20:
                            seen_lines.add(line_stripped)
                        filtered_lines.append(line)
                else:
                    consecutive_duplicates = 0
                    filtered_lines.append(line)  # Keep empty lines for formatting
            
            response_clean = '\n'.join(filtered_lines)
            
            # Limit response length to prevent excessive output
            max_response_length = 2000  # Reduced from 3000
            if len(response_clean) > max_response_length:
                # Try to cut at a sentence boundary
                truncated = response_clean[:max_response_length]
                last_period = truncated.rfind('.')
                last_newline = truncated.rfind('\n')
                cut_point = max(last_period, last_newline)
                if cut_point > max_response_length * 0.7:  # Only use if we're not cutting too much
                    response_clean = response_clean[:cut_point + 1]
                else:
                    response_clean = truncated
                
                # Don't add truncation message if we're near the end anyway
                if len(response_clean) < max_response_length - 100:
                    response_clean += "\n\n[Response truncated]"
            
            return {
                "agent": self.name,
                "role": self.role,
                "response": response_clean,
                "success": True
            }
            
        except Exception as e:
            return {
                "agent": self.name,
                "role": self.role,
                "response": f"Error during processing: {str(e)}",
                "success": False
            }
    
    def to_dict(self) -> Dict[str, str]:
        """Export agent configuration to dictionary"""
        return {
            "name": self.name,
            "role": self.role,
            "system_prompt": self._system_prompt,
            "few_shot_examples": self._few_shot_examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str], llm=None):
        """Create agent from dictionary"""
        return cls(
            name=data.get("name", ""),
            role=data.get("role", ""),
            system_prompt=data.get("system_prompt", ""),
            llm=llm,
            few_shot_examples=data.get("few_shot_examples", "")
        )


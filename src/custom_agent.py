"""
Custom Agent - User-defined agents with custom prompts
Adapted from CyberXP for general-purpose use
"""

from typing import Dict, Any, Generator
from .base_agent import BaseAgent
import re

# Pre-compile commonly used regex patterns for better performance
_URL_PATTERN = re.compile(r'https?://[^\s]+')
_CODE_BLOCK_PATTERN = re.compile(r'```[\w]*\n.*?```|```.*?```', re.DOTALL | re.MULTILINE)

_PROMPT_PATTERNS_TO_REMOVE = [
    r'Data source:.*?(?=\n|$)',
    r'Strategy:.*?(?=\n|$)',
    r'Output format:.*?(?=\n|$)',
    r'Output types:.*?(?=\n|$)',
    r'Focus on:.*?(?=\n|$)',
    r'Route:.*?(?=\n|$)',
    r'Dates:.*?(?=\n|$)',
    r'IMPORTANT:.*?(?=\n\n|\n[A-Z]|$)',
    r'Your response should.*?(?=\n\n|\n[A-Z]|$)',
    r'Do NOT include.*?(?=\n\n|\n[A-Z]|$)',
    r'Be direct.*?(?=\n|$)',
    r'Context:.*?(?=\n\n|$)',
    r'## Previous Results:.*?(?=\n\n|$)',
    r'\[.*?Agent\]:\s*',
    r'Human:.*?(?=\n|$)',
    r'\*\*Role:\*\*.*?(?=\n|$)',
    r'Role:.*?(?=\n|$)',
    r'^- .*?(?:focused on|Using|Checking|Adapting|Encouraging|Providing).*?$',
]

_PROMPT_SKIP_LINE_PATTERNS = [
    r'^Data source',
    r'^Strategy',
    r'^Output format',
    r'^Output types',
    r'^Focus on',
    r'^Route:',
    r'^Dates:',
    r'^IMPORTANT:',
    r'^Your response',
    r'^Do NOT',
    r'^Be direct',
    r'^Context:',
    r'^Human:',
    r'^## Previous',
    r'^\*\*Role:',
    r'^Role:',
    r'^- .*?(?:Breaking down|Using|Checking|Adapting|Encouraging|Providing)',
    r'^- .*?(?:focused on|clear explanations)',
    r'^Explain concepts',
]

_PROMPT_CODE_LINE_PATTERNS = [
    r'^import\s+',
    r'^from\s+\w+\s+import',
    r'^def\s+\w+\s*\(',
    r'^class\s+\w+',
    r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^=]',
    r'^\s*print\s*\(',
    r'^\s*return\s+',
    r'^\s*if\s+.*:',
    r'^\s*for\s+.*:',
    r'^\s*while\s+.*:',
    r'^\s*#.*',
]


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
            
            # Clean response: remove common artifacts and ALL guidelines/metadata
            response_clean = response.strip()
            
            # Remove any accidental prompt echoes at start
            if response_clean.startswith(("System:", "Human:", "Assistant:")):
                lines = response_clean.split('\n', 1)
                if len(lines) > 1:
                    response_clean = lines[1].strip()
            
            # Aggressive cleaning: remove guidelines, metadata, system prompts, and code blocks
            response_clean = _URL_PATTERN.sub('', response_clean)
            response_clean = _CODE_BLOCK_PATTERN.sub('', response_clean)

            for pattern in _PROMPT_PATTERNS_TO_REMOVE:
                response_clean = re.sub(pattern, '', response_clean, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
            
            lines = response_clean.split('\n')
            filtered_lines = []

            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    filtered_lines.append(line)
                    continue
                
                should_skip = False
                for pattern in _PROMPT_SKIP_LINE_PATTERNS:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        should_skip = True
                        break
                
                if not should_skip:
                    for code_pattern in _PROMPT_CODE_LINE_PATTERNS:
                        if re.match(code_pattern, line_stripped):
                            should_skip = True
                            break
                
                if not should_skip:
                    filtered_lines.append(line)
            
            response_clean = '\n'.join(filtered_lines).strip()
            
            # Remove repetitive patterns FIRST (before length check)
            # Use a sliding window approach for better efficiency
            lines = response_clean.split('\n')
            seen_lines = set()
            seen_phrases = set()  # Track longer phrases to catch repetitive content
            filtered_lines = []
            recent_lines = []  # Sliding window of recent lines (max 20)
            MAX_RECENT = 20
            
            for line in lines:
                line_stripped = line.strip()
                line_lower = line_stripped.lower()
                
                if not line_lower:
                    filtered_lines.append(line)
                    continue
                
                # Check for exact duplicates
                if line_lower in seen_lines:
                    continue  # Skip exact duplicates
                
                # Check for very similar lines (for long lines)
                is_duplicate = False
                if len(line_lower) > 50:
                    # For long lines, check if we've seen a very similar phrase
                    # Extract key phrases (first 100 chars, last 100 chars)
                    key_phrase = line_lower[:100] + "..." + line_lower[-100:] if len(line_lower) > 200 else line_lower
                    if key_phrase in seen_phrases:
                        is_duplicate = True
                    else:
                        seen_phrases.add(key_phrase)
                
                # Check similarity with recent lines only (more efficient than checking all)
                if not is_duplicate and len(line_lower) > 30:
                    for seen in recent_lines:  # Only check recent lines
                        if seen and len(seen) > 30:
                            # More aggressive similarity check
                            words_current = set(line_lower.split()[:15])
                            words_seen = set(seen.split()[:15])
                            common_words = words_current & words_seen
                            # If more than 60% of words are common, consider it duplicate
                            if len(common_words) > max(5, len(words_current) * 0.6):
                                is_duplicate = True
                                break
                
                if not is_duplicate:
                    if len(line_lower) > 10:
                        seen_lines.add(line_lower)
                        # Maintain sliding window
                        recent_lines.append(line_lower)
                        if len(recent_lines) > MAX_RECENT:
                            recent_lines.pop(0)
                    filtered_lines.append(line)
            
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
    
    def process_stream(self, input_text: str, context: str = "") -> Generator[str, None, None]:
        """
        Process input using streaming generation.
        
        Args:
            input_text: Input text to process
            context: Additional context
        
        Yields:
            Response chunks as they're generated
        """
        if not self.llm:
            yield "Error: No LLM configured for this agent"
            return
        
        try:
            from langchain_core.prompts import ChatPromptTemplate
            
            # Build prompt (same as process method)
            system_prompt = self._system_prompt.strip()
            
            max_context_length = 2000
            truncated_context = context.strip()[:max_context_length] if len(context.strip()) > max_context_length else context.strip()
            
            if truncated_context:
                user_message = f"{input_text.strip()}\n\nContext: {truncated_context}"
            else:
                user_message = input_text.strip()
            
            max_user_message = 3000
            if len(user_message) > max_user_message:
                user_message = user_message[:max_user_message] + "... [truncated]"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\n**IMPORTANT**: Your response should ONLY contain the actual answer/data. Do NOT include any guidelines, instructions, or meta-commentary in your response. Be direct and factual."),
                ("human", user_message)
            ])
            
            # Create streaming chain
            chain = prompt | self.llm
            
            # Stream response
            for chunk in chain.stream({}):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
                else:
                    yield str(chunk)
                    
        except Exception as e:
            yield f"Error during streaming: {str(e)}"
    
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


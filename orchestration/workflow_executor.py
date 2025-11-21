"""
Workflow Executor - Execute parsed workflows using AgentForge agents
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .orchestration_parser import OrchestrationParser
from src.agent_manager import AgentManager
from src.custom_agent import CustomAgent


class WorkflowExecutor:
    """Execute canvas workflows using AgentForge agents"""
    
    def __init__(self, agent_manager: AgentManager, llm=None, auto_create_agents: bool = True):
        """
        Initialize workflow executor.
        
        Args:
            agent_manager: AgentManager instance
            llm: Language model instance
            auto_create_agents: If True, create agents that don't exist from workflow config
        """
        self.agent_manager = agent_manager
        self.llm = llm
        self.auto_create_agents = auto_create_agents
        self.parser = OrchestrationParser()
    
    def execute_workflow(
        self,
        canvas_json: Dict[str, Any],
        initial_input: str,
        context: str = "",
        pass_mode: str = "cumulative"
    ) -> Dict[str, Any]:
        """
        Execute a canvas workflow.
        
        Args:
            canvas_json: Canvas workflow JSON
            initial_input: Initial input text
            context: Additional context
            pass_mode: How to pass data between agents
                - "cumulative": Each agent sees all previous outputs
                - "sequential": Each agent only sees previous agent's output
                - "parallel": All agents see only initial input
        
        Returns:
            Dictionary with execution results
        """
        try:
            # Parse workflow
            parsed_workflow = self.parser.parse_workflow(canvas_json)
            
            # Validate workflow
            is_valid, errors = self.parser.validate_workflow(parsed_workflow)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Workflow validation failed: {', '.join(errors)}",
                    "workflow_name": parsed_workflow.get("name", "Unknown")
                }
            
            # Get agent sequence
            agent_sequence = parsed_workflow.get("agent_sequence", [])
            if not agent_sequence:
                return {
                    "success": False,
                    "error": "No agents found in workflow",
                    "workflow_name": parsed_workflow.get("name", "Unknown")
                }
            
            # Ensure all agents exist
            missing_agents = []
            for agent_config in agent_sequence:
                agent_name = agent_config.get("name")
                if not self.agent_manager.agent_exists(agent_name):
                    if self.auto_create_agents:
                        # Create agent from config
                        agent = self._create_agent_from_config(agent_config)
                        if agent:
                            self.agent_manager.save_agent(agent)
                        else:
                            missing_agents.append(agent_name)
                    else:
                        missing_agents.append(agent_name)
            
            if missing_agents:
                return {
                    "success": False,
                    "error": f"Agents not found and could not be created: {', '.join(missing_agents)}",
                    "workflow_name": parsed_workflow.get("name", "Unknown")
                }
            
            # Execute agents in sequence
            results = []
            current_input = initial_input
            all_outputs = []
            
            for idx, agent_config in enumerate(agent_sequence):
                agent_name = agent_config.get("name")
                
                # Load agent
                agent = self.agent_manager.load_agent(agent_name, llm=self.llm)
                if not agent:
                    results.append({
                        "agent": agent_name,
                        "role": agent_config.get("role", "Unknown"),
                        "response": f"Error: Could not load agent '{agent_name}'",
                        "success": False,
                        "step": idx + 1
                    })
                    continue
                
                # Determine input based on pass mode
                if pass_mode == "cumulative" and idx > 0:
                    # Include all previous outputs with better formatting
                    # Truncate each response to avoid overwhelming the model
                    previous_outputs = []
                    for r in results:
                        if r.get('success', False):
                            agent_name = r.get('agent', 'Unknown')
                            response = r.get('response', '')
                            # Truncate each response to max 500 chars to keep context manageable
                            if len(response) > 500:
                                # Try to cut at sentence boundary
                                truncated = response[:500]
                                last_period = truncated.rfind('.')
                                if last_period > 400:
                                    response = response[:last_period + 1] + " [summary]"
                                else:
                                    response = truncated + " [summary]"
                            previous_outputs.append(f"**{agent_name}**: {response}")
                    
                    chain_context = "## Previous Results:\n\n" + "\n\n".join(previous_outputs)
                    # Limit total context length
                    max_context = 2000
                    if len(chain_context) > max_context:
                        chain_context = chain_context[:max_context] + "\n[Previous results truncated]"
                    
                    agent_input = initial_input
                    agent_context = f"{context}\n\n{chain_context}" if context else chain_context
                    
                    # Final truncation of total context
                    if len(agent_context) > 2500:
                        agent_context = agent_context[:2500] + "\n[Context truncated]"
                
                elif pass_mode == "sequential" and idx > 0:
                    # Only use previous agent's output, but include original context
                    prev_response = results[-1]['response'] if results else ""
                    agent_input = prev_response
                    agent_context = f"{context}\n\n**Original Request:** {initial_input}" if context else f"**Original Request:** {initial_input}"
                
                else:  # First agent or parallel mode
                    agent_input = current_input
                    agent_context = context
                
                # Process with agent
                try:
                    result_dict = agent.process(agent_input, context=agent_context)
                    result_dict["step"] = idx + 1
                    results.append(result_dict)
                    
                    if result_dict.get("success"):
                        all_outputs.append(f"[{agent_name}]: {result_dict.get('response', '')}")
                
                except Exception as e:
                    results.append({
                        "agent": agent_name,
                        "role": agent_config.get("role", "Unknown"),
                        "response": f"Error during execution: {str(e)}",
                        "success": False,
                        "step": idx + 1
                    })
            
            # Build final output
            successful_steps = sum(1 for r in results if r.get("success", False))
            final_output = "\n\n".join(all_outputs) if all_outputs else "No successful outputs"
            
            return {
                "success": successful_steps > 0,
                "workflow_name": parsed_workflow.get("name", "Unknown"),
                "workflow_notes": parsed_workflow.get("notes", ""),
                "total_steps": len(agent_sequence),
                "successful_steps": successful_steps,
                "results": results,
                "final_output": final_output,
                "metadata": parsed_workflow.get("metadata", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}",
                "workflow_name": canvas_json.get("name", "Unknown")
            }
    
    def _create_agent_from_config(self, agent_config: Dict[str, Any]) -> Optional[CustomAgent]:
        """
        Create a CustomAgent from workflow agent configuration.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            CustomAgent instance or None if creation fails
        """
        try:
            name = agent_config.get("name", "Unknown Agent")
            role = agent_config.get("role", name)
            system_prompt = agent_config.get("system_prompt", "")
            
            # If no system prompt, create a basic one
            if not system_prompt:
                system_prompt = f"You are {name}. {role}. Provide helpful, accurate responses based on the input provided."
            
            # Create agent
            agent = CustomAgent(
                name=name,
                role=role,
                system_prompt=system_prompt,
                llm=self.llm
            )
            
            return agent
            
        except Exception as e:
            print(f"Error creating agent from config: {e}")
            return None

"""
Agent Chain - Sequential execution with data passing
"""

from typing import List, Dict, Any, Optional, Literal

# Valid pass modes
PassMode = Literal["cumulative", "sequential", "parallel"]


class AgentChain:
    """Manages sequential agent execution with data passing"""
    
    def __init__(self, agent_manager, llm):
        """
        Initialize agent chain.
        
        Args:
            agent_manager: AgentManager instance
            llm: Language model instance
        """
        self.agent_manager = agent_manager
        self.llm = llm
    
    def execute_chain(
        self, 
        agent_names: List[str], 
        initial_input: str, 
        context: str = "",
        pass_mode: PassMode = "cumulative"
    ) -> Dict[str, Any]:
        """
        Execute agents in sequence.
        
        Args:
            agent_names: List of agent names in order
            initial_input: Initial input text
            context: Additional context
            pass_mode: How to pass data between agents
                - "cumulative": Each agent sees all previous outputs
                - "sequential": Each agent only sees previous agent's output
                - "parallel": All agents see only initial input (no chaining)
        
        Returns:
            Dictionary with chain results
        """
        # Validate inputs
        if not agent_names:
            return {
                "success": False,
                "error": "No agents specified",
                "results": []
            }
        
        if pass_mode not in ("cumulative", "sequential", "parallel"):
            return {
                "success": False,
                "error": f"Invalid pass_mode: '{pass_mode}'. Must be 'cumulative', 'sequential', or 'parallel'",
                "results": []
            }
        
        results = []
        current_input = initial_input
        all_outputs = []
        
        for idx, agent_name in enumerate(agent_names):
            agent = self.agent_manager.load_agent(agent_name, llm=self.llm)
            
            if not agent:
                results.append({
                    "agent": agent_name,
                    "role": "Unknown",
                    "response": f"Error: Could not load agent '{agent_name}'",
                    "success": False,
                    "step": idx + 1
                })
                continue
            
            # Determine input based on pass mode
            if pass_mode == "cumulative" and idx > 0:
                # Include all previous outputs
                chain_context = f"Previous agents' outputs:\n\n" + "\n\n".join([
                    f"[{r['agent']}]: {r['response']}" 
                    for r in results if r.get('success', False)
                ])
                agent_input = initial_input
                agent_context = f"{context}\n\n{chain_context}" if context else chain_context
            
            elif pass_mode == "sequential" and idx > 0:
                # Only use previous agent's output
                agent_input = results[-1]['response']
                agent_context = context
            
            else:  # First agent or parallel mode
                agent_input = current_input
                agent_context = context
            
            # Process with agent
            try:
                result = agent.process(agent_input, context=agent_context)
                result["step"] = idx + 1
                results.append(result)
                
                if result.get("success", False):
                    all_outputs.append(result["response"])
                    
            except Exception as e:
                results.append({
                    "agent": agent_name,
                    "role": agent.role,
                    "response": f"Error during processing: {str(e)}",
                    "success": False,
                    "step": idx + 1
                })
        
        # Generate summary
        successful_steps = sum(1 for r in results if r.get("success", False))
        
        return {
            "success": successful_steps == len(agent_names),
            "total_steps": len(agent_names),
            "successful_steps": successful_steps,
            "pass_mode": pass_mode,
            "initial_input": initial_input,
            "context": context,
            "results": results,
            "final_output": results[-1]["response"] if results else ""
        }
    
    def create_workflow(
        self,
        workflow_name: str,
        agent_sequence: List[str],
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a reusable workflow definition.
        
        Args:
            workflow_name: Name of the workflow
            agent_sequence: List of agent names in order
            description: Workflow description
        
        Returns:
            Workflow definition
        """
        return {
            "name": workflow_name,
            "description": description,
            "agents": agent_sequence,
            "agent_count": len(agent_sequence)
        }


class WorkflowPresets:
    """Pre-defined workflow templates"""
    
    @staticmethod
    def get_all_workflows() -> List[Dict[str, Any]]:
        """Get all preset workflows"""
        return [
            {
                "name": "Content Pipeline",
                "description": "Brainstorm → Write → Review → Polish",
                "agents": ["Brainstorming Assistant", "Creative Writer", "Technical Writer", "Code Reviewer"],
                "recommended_pass_mode": "cumulative"
            },
            {
                "name": "Code Review Pipeline",
                "description": "Review code → Find bugs → Suggest refactoring",
                "agents": ["Code Reviewer", "Bug Hunter", "Refactoring Expert"],
                "recommended_pass_mode": "cumulative"
            },
            {
                "name": "Research & Analysis",
                "description": "Research → Analyze → Report",
                "agents": ["Research Analyst", "Data Analyst", "Technical Writer"],
                "recommended_pass_mode": "sequential"
            },
            {
                "name": "Marketing Content",
                "description": "Strategy → Copy → Review",
                "agents": ["Business Strategist", "Marketing Copywriter", "UX Reviewer"],
                "recommended_pass_mode": "sequential"
            },
            {
                "name": "Customer Support Escalation",
                "description": "Initial support → Technical troubleshooting",
                "agents": ["Customer Support Agent", "Technical Support"],
                "recommended_pass_mode": "sequential"
            },
            {
                "name": "Security Assessment",
                "description": "Multiple security perspectives",
                "agents": ["Cybersecurity Analyst", "Code Reviewer", "Risk Analyst"],
                "recommended_pass_mode": "parallel"
            }
        ]
    
    @staticmethod
    def get_workflow_by_name(name: str) -> Optional[Dict[str, Any]]:
        """Get a specific workflow by name"""
        workflows = WorkflowPresets.get_all_workflows()
        for workflow in workflows:
            if workflow["name"] == name:
                return workflow
        return None


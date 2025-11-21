"""
Workflow Code Generator - Generate Python code from canvas workflows
"""

from typing import Dict, Any, Optional
from .orchestration_parser import OrchestrationParser


class WorkflowCodeGenerator:
    """Generate executable Python code from canvas workflows"""
    
    def __init__(self):
        """Initialize code generator"""
        self.parser = OrchestrationParser()
    
    def generate_code(
        self,
        canvas_json: Dict[str, Any],
        model_path: str = "abaryan/CyberXP_Agent_Llama_3.2_1B",
        pass_mode: str = "cumulative"
    ) -> str:
        """
        Generate Python code to execute the workflow.
        
        Args:
            canvas_json: Canvas workflow JSON
            model_path: Hugging Face model path
            pass_mode: Data passing mode
            
        Returns:
            Python code as string
        """
        try:
            # Parse workflow
            parsed = self.parser.parse_workflow(canvas_json)
            
            # Generate code
            code = self._generate_imports()
            code += self._generate_initialization(model_path)
            code += self._generate_agent_creation(parsed)
            code += self._generate_workflow_execution(parsed, pass_mode)
            code += self._generate_output()
            
            return code
            
        except Exception as e:
            return f"# Error generating code: {str(e)}"
    
    def _generate_imports(self) -> str:
        """Generate import statements"""
        return """# Generated AgentForge Orchestration Code
# This code was generated from a canvas workflow

import os
import sys
from pathlib import Path

# Add AgentForge to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

from src.agent_manager import AgentManager
from src.agent_chain import AgentChain
from src.custom_agent import CustomAgent
from src.workflow_executor import WorkflowExecutor

"""
    
    def _generate_initialization(self, model_path: str) -> str:
        """Generate initialization code"""
        return f"""# ============================================================================
# INITIALIZATION
# ============================================================================

print("Loading model: {model_path}...")
tokenizer = AutoTokenizer.from_pretrained("{model_path}", use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "{model_path}",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=800,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1,
)

llm = HuggingFacePipeline(pipeline=pipe)

# Initialize managers
agent_manager = AgentManager(storage_dir="custom_agents")
workflow_executor = WorkflowExecutor(agent_manager, llm, auto_create_agents=True)

print("Initialization complete!\\n")

"""
    
    def _generate_agent_creation(self, parsed: Dict[str, Any]) -> str:
        """Generate agent creation code"""
        nodes = parsed.get("nodes", {})
        agent_sequence = parsed.get("agent_sequence", [])
        
        code = "# ============================================================================\n"
        code += "# AGENT CREATION\n"
        code += "# ============================================================================\n\n"
        
        code += "# Agents will be auto-created if they don't exist\n"
        code += "# You can also manually create them using:\n\n"
        
        for agent_name in agent_sequence:
            # Find node for this agent
            node_data = None
            for node_id, node in nodes.items():
                template = node.get("template", "")
                title = node.get("title", "")
                if (OrchestrationParser.TEMPLATE_TO_AGENT.get(template) == agent_name or 
                    title == agent_name):
                    node_data = node
                    break
            
            if node_data:
                config = node_data.get("config", {})
                role = config.get("role", config.get("Role", "AI Agent"))
                template_id = node_data.get("template", "")
                
                code += f"""# Agent: {agent_name}
# agent = CustomAgent(
#     name="{agent_name}",
#     role="{role}",
#     system_prompt="Your system prompt here",
#     llm=llm
# )
# agent_manager.save_agent(agent)

"""
        
        return code
    
    def _generate_workflow_execution(self, parsed: Dict[str, Any], pass_mode: str) -> str:
        """Generate workflow execution code"""
        workflow_name = parsed.get("name", "Untitled Workflow")
        workflow_notes = parsed.get("notes", "")
        agent_sequence = parsed.get("agent_sequence", [])
        
        code = "# ============================================================================\n"
        code += "# WORKFLOW EXECUTION\n"
        code += "# ============================================================================\n\n"
        
        code += f"""# Workflow: {workflow_name}
# Notes: {workflow_notes}
# Agents: {', '.join(agent_sequence)}
# Pass Mode: {pass_mode}

# Canvas workflow JSON
canvas_workflow = {self._format_json(parsed)}

# Execute workflow
initial_input = "Your input text here"
context = "Additional context (optional)"

result = workflow_executor.execute_workflow(
    canvas_json=canvas_workflow,
    initial_input=initial_input,
    context=context,
    pass_mode="{pass_mode}"
)

"""
        
        return code
    
    def _generate_output(self) -> str:
        """Generate output handling code"""
        return """# ============================================================================
# OUTPUT
# ============================================================================

if result.get("success"):
    print("\\n✅ Workflow executed successfully!\\n")
    print(f"Total steps: {result['total_steps']}")
    print(f"Successful steps: {result['successful_steps']}\\n")
    
    print("Results:")
    for step_result in result.get("results", []):
        agent = step_result.get("agent", "Unknown")
        response = step_result.get("response", "")
        success = step_result.get("success", False)
        status = "✅" if success else "❌"
        
        print(f"\\n{status} {agent}:")
        print(f"{response[:200]}..." if len(response) > 200 else response)
    
    print(f"\\n📊 Final Output:\\n{result.get('final_output', '')}")
else:
    print(f"\\n❌ Workflow execution failed: {result.get('error', 'Unknown error')}")
    if result.get("errors"):
        for error in result["errors"]:
            print(f"  - {error}")

"""
    
    def _format_json(self, data: Dict[str, Any]) -> str:
        """Format dictionary as JSON string for code"""
        import json
        # Only include essential data
        simplified = {
            "name": data.get("name", ""),
            "notes": data.get("notes", ""),
            "graph": {
                "cells": []  # Simplified - actual execution uses parsed data
            }
        }
        return json.dumps(simplified, indent=2)
    
    def generate_simple_code(
        self,
        canvas_json: Dict[str, Any],
        model_path: str = "abaryan/CyberXP_Agent_Llama_3.2_1B"
    ) -> str:
        """
        Generate simplified code using AgentChain directly.
        
        Args:
            canvas_json: Canvas workflow JSON
            model_path: Hugging Face model path
            
        Returns:
            Simplified Python code
        """
        parsed = self.parser.parse_workflow(canvas_json)
        agent_sequence = parsed.get("agent_sequence", [])
        workflow_name = parsed.get("name", "Untitled Workflow")
        
        code = f"""# Simple AgentForge Workflow Execution
# Workflow: {workflow_name}

from src.agent_manager import AgentManager
from src.agent_chain import AgentChain
# ... (add your imports and initialization)

# Agent sequence from workflow
agent_names = {agent_sequence}

# Execute chain
agent_chain = AgentChain(agent_manager, llm)
result = agent_chain.execute_chain(
    agent_names=agent_names,
    initial_input="Your input here",
    context="Additional context",
    pass_mode="cumulative"
)

print(result["final_output"])
"""
        
        return code


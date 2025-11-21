"""
AgentForge - Universal Multi-Agent Manager
Main Gradio application for creating, managing, and orchestrating AI agents
"""

import os
import sys
from typing import List, Tuple, Optional, Dict, Any

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# Ensure src directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.custom_agent import CustomAgent
from src.agent_manager import AgentManager
from src.agent_templates import AgentTemplates
from src.history_manager import HistoryManager
from src.agent_chain import AgentChain, WorkflowPresets
from src.rag_integration import SimpleRAG

# Orchestration imports
import json
from orchestration.workflow_executor import WorkflowExecutor
from orchestration.orchestration_parser import OrchestrationParser

# Configuration
DEFAULT_MODEL = "abaryan/CyberXP_Agent_Llama_3.2_1B"

# Global instances
llm = None
agent_manager = None
history_manager = None
agent_chain = None
rag_system = None
workflow_executor = None


def initialize(model_path: str = DEFAULT_MODEL):
    """
    Initialize LLM and all manager instances.
    
    Args:
        model_path: Hugging Face model identifier or local path
    """
    global llm, agent_manager, history_manager, agent_chain, rag_system
    
    print(f"Loading model: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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
    
    # Initialize all managers
    agent_manager = AgentManager(storage_dir="custom_agents")
    history_manager = HistoryManager(history_dir="history")
    agent_chain = AgentChain(agent_manager, llm)
    rag_system = SimpleRAG(knowledge_base_dir="knowledge_base")
    
    # Initialize workflow executor (needs agent_manager and llm)
    global workflow_executor
    workflow_executor = WorkflowExecutor(agent_manager, llm, auto_create_agents=True)
    
    print("Initialization complete!")


# ============================================================================
# AGENT CREATION AND EDITING
# ============================================================================

def create_agent_handler(name: str, role: str, prompt: str) -> Tuple[str, gr.update, gr.update, gr.update]:
    """
    Create a new custom agent.
    
    Args:
        name: Agent name
        role: Agent role/specialty
        prompt: System prompt defining behavior
    
    Returns:
        Tuple of (status message, agent_selector update, chain_agents update, edit_selector update)
    """
    try:
        if not name or not role or not prompt:
            return "⚠️ Please fill in all fields", gr.update(), gr.update(), gr.update()
        
        if agent_manager.agent_exists(name):
            return f"⚠️ Agent '{name}' already exists", gr.update(), gr.update(), gr.update()
        
        agent = CustomAgent(
            name=name.strip(),
            role=role.strip(),
            system_prompt=prompt.strip(),
            llm=llm
        )
        
        success = agent_manager.save_agent(agent)
        if success:
            updated_choices = [agent["name"] for agent in agent_manager.list_agents()]
            return (
                f"✅ Agent '{name}' created successfully!", 
                gr.update(choices=updated_choices),
                gr.update(choices=updated_choices),
                gr.update(choices=updated_choices)
            )
        else:
            return "❌ Failed to save agent", gr.update(), gr.update(), gr.update()
    
    except ValueError as e:
        return f"❌ Validation error: {str(e)}", gr.update(), gr.update(), gr.update()
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(), gr.update(), gr.update()


def load_template(template_name: str) -> Tuple[str, str, str]:
    """
    Load a template into the agent creation form.
    
    Args:
        template_name: Name of template to load
    
    Returns:
        Tuple of (name, role, system_prompt)
    """
    if not template_name:
        return "", "", ""
    
    template = AgentTemplates.get_template_by_name(template_name)
    if template:
        return template["name"], template["role"], template["system_prompt"]
    return "", "", ""


def load_agent_for_edit(agent_name: str) -> Tuple[str, str, str, str]:
    """Load agent data for editing"""
    if not agent_name:
        return "", "", "", ""
    
    agent_data = agent_manager.get_agent_data(agent_name)
    if agent_data:
        return (
            agent_data.get("name", ""),
            agent_data.get("role", ""),
            agent_data.get("system_prompt", ""),
            agent_name  # Store original name
        )
    return "", "", "", ""


def update_agent_handler(
    original_name: str,
    new_name: str,
    role: str,
    prompt: str
) -> Tuple[str, gr.update, gr.update, gr.update]:
    """Update an existing agent"""
    try:
        if not original_name:
            return "⚠️ No agent selected for editing", gr.update(), gr.update(), gr.update()
        
        if not new_name or not role or not prompt:
            return "⚠️ Please fill in all fields", gr.update(), gr.update(), gr.update()
        
        # Check if renaming to an existing agent
        if new_name != original_name and agent_manager.agent_exists(new_name):
            return f"⚠️ Agent '{new_name}' already exists", gr.update(), gr.update(), gr.update()
        
        agent = CustomAgent(
            name=new_name.strip(),
            role=role.strip(),
            system_prompt=prompt.strip(),
            llm=llm
        )
        
        success = agent_manager.update_agent(original_name, agent)
        if success:
            updated_choices = [agent["name"] for agent in agent_manager.list_agents()]
            return (
                f"✅ Agent '{new_name}' updated successfully!",
                gr.update(choices=updated_choices),
                gr.update(choices=updated_choices),
                gr.update(choices=updated_choices)
            )
        else:
            return "❌ Failed to update agent", gr.update(), gr.update(), gr.update()
    
    except ValueError as e:
        return f"❌ Validation error: {str(e)}", gr.update(), gr.update(), gr.update()
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(), gr.update(), gr.update()


# ============================================================================
# AGENT EXECUTION
# ============================================================================

def run_agents(
    input_text: str, 
    context: str, 
    selected_agents: List[str],
    use_rag: bool = False
) -> Tuple[str, str]:
    """
    Execute selected agents on input text.
    
    Args:
        input_text: User input text
        context: Additional context
        selected_agents: List of agent names to run
        use_rag: Whether to use knowledge base retrieval
    
    Returns:
        Tuple of (text_output, html_output)
    """
    if not input_text.strip():
        return "Please provide input text", ""
    
    if not selected_agents or len(selected_agents) == 0:
        return "Please select at least one agent", ""
    
    # Add RAG context if enabled
    if use_rag:
        rag_context = rag_system.get_context_for_query(input_text, top_k=3)
        context = f"{context}\n\n{rag_context}" if context else rag_context
    
    results = []
    agent_names = []
    responses_dict = {}
    
    for agent_name in selected_agents:
        agent = agent_manager.load_agent(agent_name, llm=llm)
        
        if agent:
            result_dict = agent.process(input_text.strip(), context=context)
            response = result_dict.get("response", "")
            
            results.append(f"## {agent_name}\n**Role:** {result_dict.get('role', '')}\n\n{response}")
            agent_names.append(agent_name)
            responses_dict[agent_name] = response
        else:
            results.append(f"## {agent_name}\n\n❌ Error: Could not load agent")
            agent_names.append(f"{agent_name} (error)")
            responses_dict[agent_name] = "Error: Could not load agent"
    
    # Save to history
    history_manager.save_conversation(
        input_text=input_text,
        context=context,
        agents_used=selected_agents,
        responses=responses_dict
    )
    
    # Combine all results
    combined_text = "\n\n---\n\n".join(results)
    
    # Create HTML output
    html_output = "<div style='padding: 20px; background: #1a1f26; color: #e6edf3;'>"
    
    for idx, agent_name in enumerate(agent_names):
        border_color = ["#8ab4f8", "#ffa500", "#50fa7b", "#ff79c6", "#bd93f9"][idx % 5]
        html_output += f"""
        <div style='margin-bottom: 30px; border-left: 5px solid {border_color}; padding: 20px; background: #0d1117; border-radius: 8px;'>
            <h2 style='color: {border_color}; margin-bottom: 15px;'>🤖 {agent_name}</h2>
            <div style='white-space: pre-wrap; line-height: 1.6;'>{results[idx].split('\\n\\n', 2)[-1]}</div>
        </div>
        """
    
    html_output += "</div>"
    
    return combined_text, html_output


def run_agent_chain(
    input_text: str,
    context: str,
    selected_agents: List[str],
    pass_mode: str
) -> Tuple[str, str]:
    """
    Execute agents sequentially in a chain.
    
    Args:
        input_text: Initial input text
        context: Additional context
        selected_agents: List of agent names in execution order
        pass_mode: Data passing mode (cumulative/sequential/parallel)
    
    Returns:
        Tuple of (text_output, html_output)
    """
    if not input_text.strip():
        return "Please provide input text", ""
    
    if not selected_agents or len(selected_agents) < 2:
        return "Please select at least 2 agents for chaining", ""
    
    result = agent_chain.execute_chain(
        agent_names=selected_agents,
        initial_input=input_text,
        context=context,
        pass_mode=pass_mode
    )
    
    # Format results
    text_parts = [f"# Agent Chain Execution\n"]
    text_parts.append(f"**Mode:** {pass_mode}")
    text_parts.append(f"**Steps:** {result['total_steps']}")
    text_parts.append(f"**Successful:** {result['successful_steps']}\n")
    
    html_parts = ["<div style='padding: 20px; background: #1a1f26; color: #e6edf3;'>"]
    html_parts.append(f"<div style='margin-bottom: 20px;'><strong>Mode:</strong> {pass_mode} | <strong>Steps:</strong> {result['total_steps']} | <strong>Success:</strong> {result['successful_steps']}/{result['total_steps']}</div>")
    
    for step_result in result['results']:
        step_num = step_result['step']
        agent_name = step_result['agent']
        response = step_result['response']
        success = step_result.get('success', False)
        
        status_icon = "✅" if success else "❌"
        text_parts.append(f"\n## Step {step_num}: {agent_name} {status_icon}\n{response}\n")
        
        border_color = "#50fa7b" if success else "#ff5555"
        html_parts.append(f"""
        <div style='margin-bottom: 20px; border-left: 5px solid {border_color}; padding: 15px; background: #0d1117; border-radius: 8px;'>
            <h3 style='color: {border_color};'>Step {step_num}: {agent_name} {status_icon}</h3>
            <div style='white-space: pre-wrap; line-height: 1.6; margin-top: 10px;'>{response}</div>
        </div>
        """)
    
    html_parts.append("</div>")
    
    return "".join(text_parts), "".join(html_parts)


# ============================================================================
# AGENT MANAGEMENT
# ============================================================================

def delete_agent_handler(agent_name: str) -> Tuple[str, gr.update, gr.update, gr.update]:
    """Delete an agent"""
    if not agent_name:
        return "Please select an agent to delete", gr.update(), gr.update(), gr.update()
    
    success = agent_manager.delete_agent(agent_name)
    if success:
        updated_choices = [agent["name"] for agent in agent_manager.list_agents()]
        return (
            f"✅ Agent '{agent_name}' deleted successfully",
            gr.update(choices=updated_choices),
            gr.update(choices=updated_choices),
            gr.update(choices=updated_choices)
        )
    else:
        return f"❌ Failed to delete agent '{agent_name}'", gr.update(), gr.update(), gr.update()


def refresh_agent_list() -> Tuple[str, gr.update, gr.update, gr.update]:
    """Refresh the agent list"""
    agents = agent_manager.list_agents()
    list_text = "\n\n".join([f"**{agent['name']}** - {agent['role']}" for agent in agents]) or "No agents created yet."
    updated_choices = [agent["name"] for agent in agents]
    
    return (
        list_text,
        gr.update(choices=updated_choices),
        gr.update(choices=updated_choices),
        gr.update(choices=updated_choices)
    )


def export_agent_handler(agent_name: str) -> Tuple[str, Optional[str]]:
    """Export an agent to file"""
    if not agent_name:
        return "Please select an agent to export", None
    
    output_path = f"exported_{agent_name.replace(' ', '_')}.json"
    success = agent_manager.export_agent(agent_name, output_path)
    
    if success:
        return f"✅ Agent exported to {output_path}", output_path
    return f"❌ Failed to export agent", None


def import_agent_handler(file_path: str, overwrite: bool) -> Tuple[str, gr.update, gr.update, gr.update]:
    """Import an agent from file"""
    if not file_path:
        return "Please select a file", gr.update(), gr.update(), gr.update()
    
    success = agent_manager.import_agent(file_path, llm=llm, overwrite=overwrite)
    
    if success:
        updated_choices = [agent["name"] for agent in agent_manager.list_agents()]
        return (
            "✅ Agent imported successfully",
            gr.update(choices=updated_choices),
            gr.update(choices=updated_choices),
            gr.update(choices=updated_choices)
        )
    return "❌ Failed to import agent", gr.update(), gr.update(), gr.update()


def export_all_handler() -> Tuple[str, Optional[str]]:
    """Export all agents"""
    output_path = "all_agents_backup.zip"
    success = agent_manager.export_all_agents(output_path)
    
    if success:
        return f"✅ All agents exported to {output_path}", output_path
    return "❌ Failed to export agents", None


# ============================================================================
# HISTORY AND STATISTICS
# ============================================================================

def get_history_display() -> str:
    """Get formatted history display"""
    conversations = history_manager.list_conversations(limit=20)
    
    if not conversations:
        return "No conversation history yet."
    
    parts = ["# Recent Conversations\n"]
    for conv in conversations:
        parts.append(f"**{conv['date']} {conv['time']}** - {conv['agent_count']} agent(s)")
        parts.append(f"*Input:* {conv['input_preview']}")
        parts.append(f"*Agents:* {', '.join(conv['agents_used'])}\n")
    
    return "\n".join(parts)


def get_statistics_display() -> str:
    """Get statistics display"""
    stats = history_manager.get_statistics()
    
    parts = ["# Usage Statistics\n"]
    parts.append(f"**Total Conversations:** {stats['total_conversations']}")
    parts.append(f"**Total Agent Invocations:** {stats['total_agent_invocations']}")
    parts.append(f"**Average Agents per Conversation:** {stats['average_agents_per_conversation']}")
    parts.append(f"**Unique Agents Used:** {stats['unique_agents_used']}\n")
    
    if stats['most_used_agents']:
        parts.append("## Most Used Agents\n")
        for item in stats['most_used_agents'][:10]:
            parts.append(f"- **{item['agent']}**: {item['count']} times")
    
    return "\n".join(parts)


# ============================================================================
# RAG MANAGEMENT
# ============================================================================

def add_knowledge_doc(title: str, content: str) -> str:
    """Add document to knowledge base"""
    if not title or not content:
        return "⚠️ Please provide both title and content"
    
    success = rag_system.add_document(title, content)
    if success:
        return f"✅ Document '{title}' added to knowledge base"
    return "❌ Failed to add document"


def get_kb_stats() -> str:
    """Get knowledge base statistics"""
    stats = rag_system.get_stats()
    docs = rag_system.list_documents()
    
    parts = ["# Knowledge Base\n"]
    parts.append(f"**Total Documents:** {stats['total_documents']}")
    parts.append(f"**Total Characters:** {stats['total_characters']:,}")
    parts.append(f"\n## Document Types")
    for doc_type, count in stats['document_types'].items():
        parts.append(f"- {doc_type}: {count}")
    
    if docs:
        parts.append("\n## Documents\n")
        for doc in docs:
            parts.append(f"- **{doc['title']}** ({doc['type']}) - {doc['content_length']} chars")
    
    return "\n".join(parts)


# ============================================================================
# CANVAS WORKFLOW EXECUTION
# ============================================================================

def parse_workflow_json(workflow_json_str: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse and validate canvas workflow JSON.
    
    Args:
        workflow_json_str: JSON string of canvas workflow
        
    Returns:
        Tuple of (status_message, parsed_workflow_dict)
    """
    try:
        if not workflow_json_str or not workflow_json_str.strip():
            return "⚠️ Please provide workflow JSON", None
        
        # Parse JSON
        workflow_json = json.loads(workflow_json_str)
        
        # Parse workflow
        parser = OrchestrationParser()
        parsed_workflow = parser.parse_workflow(workflow_json)
        
        # Validate
        is_valid, errors = parser.validate_workflow(parsed_workflow)
        if not is_valid:
            return f"❌ Workflow validation failed: {', '.join(errors)}", None
        
        # Build info message
        info_parts = [
            f"✅ **Workflow Parsed Successfully!**\n",
            f"**Name:** {parsed_workflow['name']}",
            f"**Version:** {parsed_workflow.get('version', 'N/A')}",
            f"**Agents:** {len(parsed_workflow['agent_sequence'])}",
            f"**Nodes:** {parsed_workflow['metadata']['node_count']}",
            f"\n### Agent Sequence:",
        ]
        
        for idx, agent_config in enumerate(parsed_workflow['agent_sequence'], 1):
            info_parts.append(f"{idx}. **{agent_config['name']}** - {agent_config['role']}")
        
        if parsed_workflow.get('notes'):
            info_parts.append(f"\n### Notes:\n{parsed_workflow['notes']}")
        
        return "\n".join(info_parts), parsed_workflow
        
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {str(e)}", None
    except Exception as e:
        return f"❌ Error parsing workflow: {str(e)}", None


def execute_workflow_handler(
    workflow_json_str: str,
    input_text: str,
    context: str,
    pass_mode: str
) -> Tuple[str, str]:
    """
    Execute a canvas workflow.
    
    Args:
        workflow_json_str: JSON string of canvas workflow
        input_text: Initial input text
        context: Additional context
        pass_mode: Data passing mode
        
    Returns:
        Tuple of (text_output, html_output)
    """
    try:
        if not workflow_json_str or not workflow_json_str.strip():
            return "⚠️ Please provide workflow JSON", ""
        
        if not input_text or not input_text.strip():
            return "⚠️ Please provide input text", ""
        
        # Parse JSON
        workflow_json = json.loads(workflow_json_str)
        
        # Execute workflow
        result = workflow_executor.execute_workflow(
            canvas_json=workflow_json,
            initial_input=input_text.strip(),
            context=context.strip() if context else "",
            pass_mode=pass_mode
        )
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            return f"❌ Workflow execution failed: {error_msg}", ""
        
        # Build text output
        text_parts = [
            f"# Workflow: {result['workflow_name']}\n",
            f"**Status:** ✅ Success",
            f"**Steps:** {result['successful_steps']}/{result['total_steps']}\n",
            "---\n"
        ]
        
        for step_result in result.get("results", []):
            agent = step_result.get("agent", "Unknown")
            role = step_result.get("role", "")
            response = step_result.get("response", "")
            success = step_result.get("success", False)
            step_num = step_result.get("step", 0)
            
            status = "✅" if success else "❌"
            text_parts.append(f"## {status} Step {step_num}: {agent}")
            if role:
                text_parts.append(f"**Role:** {role}\n")
            text_parts.append(f"{response}\n")
            text_parts.append("---\n")
        
        text_parts.append(f"\n### Final Output:\n{result.get('final_output', '')}")
        
        # Build HTML output
        html_output = "<div style='padding: 20px; background: #1a1f26; color: #e6edf3;'>"
        html_output += f"<h1 style='color: #8ab4f8; margin-bottom: 20px;'>🎯 {result['workflow_name']}</h1>"
        html_output += f"<p><strong>Status:</strong> ✅ Success | <strong>Steps:</strong> {result['successful_steps']}/{result['total_steps']}</p>"
        html_output += "<hr style='border-color: #30363d; margin: 20px 0;'>"
        
        colors = ["#8ab4f8", "#ffa500", "#50fa7b", "#ff79c6", "#bd93f9"]
        
        for idx, step_result in enumerate(result.get("results", [])):
            agent = step_result.get("agent", "Unknown")
            role = step_result.get("role", "")
            response = step_result.get("response", "")
            success = step_result.get("success", False)
            step_num = step_result.get("step", 0)
            
            border_color = colors[idx % len(colors)]
            status_icon = "✅" if success else "❌"
            
            html_output += f"""
            <div style='margin-bottom: 30px; border-left: 5px solid {border_color}; padding: 20px; background: #0d1117; border-radius: 8px;'>
                <h2 style='color: {border_color}; margin-bottom: 10px;'>{status_icon} Step {step_num}: {agent}</h2>
                {f'<p style="color: #9aa4ad; margin-bottom: 15px;"><em>{role}</em></p>' if role else ''}
                <div style='white-space: pre-wrap; line-height: 1.6;'>{response}</div>
            </div>
            """
        
        html_output += f"""
        <div style='margin-top: 30px; padding: 20px; background: #0d1117; border-radius: 8px; border: 2px solid #8ab4f8;'>
            <h2 style='color: #8ab4f8; margin-bottom: 15px;'>📊 Final Output</h2>
            <div style='white-space: pre-wrap; line-height: 1.6;'>{result.get('final_output', '')}</div>
        </div>
        """
        
        html_output += "</div>"
        
        # Save to history
        history_manager.save_conversation(
            input_text=input_text,
            context=context,
            agents_used=[r.get("agent") for r in result.get("results", [])],
            responses={r.get("agent"): r.get("response", "") for r in result.get("results", [])}
        )
        
        return "\n".join(text_parts), html_output
        
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {str(e)}", ""
    except Exception as e:
        return f"❌ Error executing workflow: {str(e)}", ""


# ============================================================================
# BUILD GRADIO INTERFACE
# ============================================================================

def build_interface():
    """
    Build the comprehensive Gradio web interface.
    
    Returns:
        Gradio Blocks app instance
    """
    
    with gr.Blocks(title="AgentForge Pro - Multi-Agent Manager", theme=gr.themes.Soft(primary_hue="blue")) as app:
        gr.Markdown("""
        # 🔨 AgentForge Pro - Universal Multi-Agent Manager
        
        Create, manage, and orchestrate multiple AI agents with advanced features:
        **Templates** • **Agent Chaining** • **RAG Integration** • **History Tracking** • **Export/Import**
        """)
        
        with gr.Tabs():
            # ================================================================
            # TAB 1: RUN AGENTS
            # ================================================================
            with gr.Tab("🚀 Run Agents"):
                gr.Markdown("### Execute Your Agents")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        input_text = gr.Textbox(
                            label="Input Text",
                            lines=5,
                            placeholder="Enter your query, question, or problem..."
                        )
                        
                        context_input = gr.Textbox(
                            label="Context (Optional)",
                            lines=3,
                            placeholder="Provide additional context if needed..."
                        )
                        
                        with gr.Row():
                            use_rag = gr.Checkbox(label="Use Knowledge Base (RAG)", value=False)
                        
                        agent_selector = gr.Dropdown(
                            choices=[agent["name"] for agent in agent_manager.list_agents()],
                            value=[],
                            label="Select Agents (Multi-Select)",
                            info="Select one or more agents",
                            multiselect=True,
                            interactive=True
                        )
                        
                        run_btn = gr.Button("▶️ Run Selected Agents", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Quick Info")
                        gr.Markdown("""
                        **Features:**
                        - Select multiple agents for diverse perspectives
                        - Enable RAG for knowledge-enhanced responses
                        - All runs are saved to history
                        
                        **Tips:**
                        - Use context for domain-specific information
                        - Combine complementary agents
                        - Check history for past runs
                        """)
                
                with gr.Tabs():
                    with gr.Tab("Visual Output"):
                        html_output = gr.HTML(
                            label="Agent Responses",
                            value='<div style="text-align:center;padding:40px;color:#9aa4ad;">Select agents and click "Run" to generate responses...</div>'
                        )
                    
                    with gr.Tab("📝 Raw Text"):
                        text_output = gr.Textbox(label="Combined Output", lines=20, placeholder="Responses will appear here...")
                
                def show_loading():
                    loading_html = '''
                    <div style="text-align:center;padding:60px;background:#11161d;border-radius:8px;">
                        <div style="font-size:48px;margin-bottom:20px;">⏳</div>
                        <div style="font-size:18px;color:#8ab4f8;margin-bottom:10px;">Running Agents...</div>
                        <div style="font-size:14px;color:#9aa4ad;">Processing your request with selected agents</div>
                        <div style="margin-top:20px;">
                            <div style="width:200px;height:4px;background:#1f2a35;margin:0 auto;border-radius:2px;overflow:hidden;">
                                <div style="width:100%;height:100%;background:linear-gradient(90deg,#8ab4f8,#6a94f8,#8ab4f8);
                                            animation:loading 1.5s ease-in-out infinite;"></div>
                            </div>
                        </div>
                        <style>
                            @keyframes loading {
                                0% { transform: translateX(-100%); }
                                100% { transform: translateX(100%); }
                            }
                        </style>
                    </div>
                    '''
                    return "", loading_html
                
                run_btn.click(
                    fn=show_loading,
                    inputs=None,
                    outputs=[text_output, html_output]
                ).then(
                    fn=run_agents,
                    inputs=[input_text, context_input, agent_selector, use_rag],
                    outputs=[text_output, html_output]
                )
            
            # ================================================================
            # TAB 2: AGENT CHAINS
            # ================================================================
            with gr.Tab("🔗 Agent Chains"):
                gr.Markdown("### Sequential Agent Execution")
                gr.Markdown("Chain multiple agents together for complex workflows")
                
                with gr.Row():
                    with gr.Column():
                        chain_input = gr.Textbox(
                            label="Input Text",
                            lines=4,
                            placeholder="Enter initial input..."
                        )
                        
                        chain_context = gr.Textbox(
                            label="Context (Optional)",
                            lines=2
                        )
                        
                        chain_agents = gr.Dropdown(
                            choices=[agent["name"] for agent in agent_manager.list_agents()],
                            value=[],
                            label="Select Agents (in order)",
                            multiselect=True,
                            interactive=True
                        )
                        
                        chain_mode = gr.Radio(
                            choices=["cumulative", "sequential", "parallel"],
                            value="cumulative",
                            label="Chain Mode",
                            info="cumulative: All previous outputs | sequential: Only previous output | parallel: No chaining"
                        )
                        
                        chain_btn = gr.Button("⚡ Run Chain", variant="primary", size="lg")
                    
                    with gr.Column():
                        gr.Markdown("### Preset Workflows")
                        workflows = WorkflowPresets.get_all_workflows()
                        workflow_info = "\n\n".join([
                            f"**{w['name']}**\n{w['description']}\nAgents: {', '.join(w['agents'][:3])}..."
                            for w in workflows[:5]
                        ])
                        gr.Markdown(workflow_info)
                
                with gr.Tabs():
                    with gr.Tab("📊 Visual"):
                        chain_html_output = gr.HTML(
                            value='<div style="text-align:center;padding:40px;color:#9aa4ad;">Select agents and click "Run Chain" to start...</div>'
                        )
                    with gr.Tab("Text"):
                        chain_text_output = gr.Textbox(lines=20, placeholder="Chain results will appear here...")
                
                def show_chain_loading():
                    loading_html = '''
                    <div style="text-align:center;padding:60px;background:#11161d;border-radius:8px;">
                        <div style="font-size:48px;margin-bottom:20px;">🔗</div>
                        <div style="font-size:18px;color:#8ab4f8;margin-bottom:10px;">Running Agent Chain...</div>
                        <div style="font-size:14px;color:#9aa4ad;">Processing sequential agent execution</div>
                        <div style="margin-top:20px;">
                            <div style="width:200px;height:4px;background:#1f2a35;margin:0 auto;border-radius:2px;overflow:hidden;">
                                <div style="width:100%;height:100%;background:linear-gradient(90deg,#8ab4f8,#6a94f8,#8ab4f8);
                                            animation:loading 1.5s ease-in-out infinite;"></div>
                            </div>
                        </div>
                        <style>
                            @keyframes loading {
                                0% { transform: translateX(-100%); }
                                100% { transform: translateX(100%); }
                            }
                        </style>
                    </div>
                    '''
                    return "", loading_html
                
                chain_btn.click(
                    fn=show_chain_loading,
                    inputs=None,
                    outputs=[chain_text_output, chain_html_output]
                ).then(
                    fn=run_agent_chain,
                    inputs=[chain_input, chain_context, chain_agents, chain_mode],
                    outputs=[chain_text_output, chain_html_output]
                )
            
            # ================================================================
            # TAB 3: CREATE AGENT
            # ================================================================
            with gr.Tab("➕ Create Agent"):
                gr.Markdown("### Create a New Agent")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### Agent Details")
                        
                        with gr.Row():
                            agent_name_input = gr.Textbox(
                                label="Agent Name",
                                placeholder="e.g., Creative Writer, Code Reviewer",
                                scale=2
                            )
                            agent_role_input = gr.Textbox(
                                label="Role/Specialty",
                                placeholder="e.g., Expert in creative storytelling",
                                scale=3
                            )
                        
                        system_prompt_input = gr.Textbox(
                            label="System Prompt",
                            placeholder="Define the agent's behavior, expertise, and output style...",
                            lines=12,
                            max_lines=20
                        )
                        
                        create_status = gr.Markdown("")
                        create_btn = gr.Button("➕ Create Agent", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### Quick Start with Templates")
                        
                        template_category = gr.Dropdown(
                            choices=AgentTemplates.get_categories(),
                            label="Category",
                            interactive=True
                        )
                        
                        template_selector = gr.Dropdown(
                            choices=[t["name"] for t in AgentTemplates.get_all_templates()],
                            label="Template",
                            interactive=True
                        )
                        
                        load_template_btn = gr.Button("📋 Load Template", size="sm")
                        
                        gr.Markdown("""
                        ### Tips
                        - Use templates as starting points
                        - Customize prompts for your needs
                        - Be specific about expected output
                        - Test and iterate
                        """)
                
                load_template_btn.click(
                    fn=load_template,
                    inputs=[template_selector],
                    outputs=[agent_name_input, agent_role_input, system_prompt_input]
                )
                
                create_btn.click(
                    fn=create_agent_handler,
                    inputs=[agent_name_input, agent_role_input, system_prompt_input],
                    outputs=[create_status, agent_selector, chain_agents, gr.State()]
                )
            
            # ================================================================
            # TAB 4: MANAGE AGENTS
            # ================================================================
            with gr.Tab("⚙️ Manage Agents"):
                gr.Markdown("### Agent Management")
                
                with gr.Tabs():
                    # Edit Agent
                    with gr.Tab("✏️ Edit Agent"):
                        gr.Markdown("#### Edit Existing Agent")
                        
                        edit_agent_selector = gr.Dropdown(
                            choices=[agent["name"] for agent in agent_manager.list_agents()],
                            label="Select Agent to Edit",
                            interactive=True
                        )
                        
                        load_edit_btn = gr.Button("📥 Load Agent", size="sm")
                        
                        edit_original_name = gr.State("")
                        
                        with gr.Row():
                            edit_name_input = gr.Textbox(label="Agent Name", scale=2)
                            edit_role_input = gr.Textbox(label="Role", scale=3)
                        
                        edit_prompt_input = gr.Textbox(label="System Prompt", lines=12)
                        
                        edit_status = gr.Markdown("")
                        
                        with gr.Row():
                            update_btn = gr.Button("💾 Update Agent", variant="primary")
                            cancel_btn = gr.Button("❌ Cancel")
                        
                        load_edit_btn.click(
                            fn=load_agent_for_edit,
                            inputs=[edit_agent_selector],
                            outputs=[edit_name_input, edit_role_input, edit_prompt_input, edit_original_name]
                        )
                        
                        update_btn.click(
                            fn=update_agent_handler,
                            inputs=[edit_original_name, edit_name_input, edit_role_input, edit_prompt_input],
                            outputs=[edit_status, agent_selector, chain_agents, edit_agent_selector]
                        )
                    
                    # View All Agents
                    with gr.Tab("📋 View All"):
                        refresh_btn = gr.Button("🔄 Refresh Agent List", size="sm")
                        
                        agent_list_display = gr.Markdown(
                            "\n\n".join([f"**{agent['name']}** - {agent['role']}" for agent in agent_manager.list_agents()]) or "No agents created yet."
                        )
                        
                        refresh_btn.click(
                            fn=refresh_agent_list,
                            outputs=[agent_list_display, agent_selector, chain_agents, edit_agent_selector]
                        )
                    
                    # Delete Agent
                    with gr.Tab("🗑️ Delete"):
                        gr.Markdown("#### Delete Agent")
                        gr.Markdown("This action cannot be undone!")
                        
                        delete_agent_selector = gr.Dropdown(
                            choices=[agent["name"] for agent in agent_manager.list_agents()],
                            label="Select Agent to Delete",
                            interactive=True
                        )
                        
                        delete_status = gr.Markdown("")
                        delete_btn = gr.Button("🗑️ Delete Agent", variant="stop")
                        
                        delete_btn.click(
                            fn=delete_agent_handler,
                            inputs=[delete_agent_selector],
                            outputs=[delete_status, agent_selector, chain_agents, edit_agent_selector]
                        )
                    
                    # Import/Export
                    with gr.Tab("📦 Import/Export"):
                        gr.Markdown("#### Export Agent")
                        
                        export_agent_selector = gr.Dropdown(
                            choices=[agent["name"] for agent in agent_manager.list_agents()],
                            label="Select Agent to Export"
                        )
                        
                        export_status = gr.Markdown("")
                        export_btn = gr.Button("📤 Export Agent")
                        export_file = gr.File(label="Exported File", interactive=False)
                        
                        export_btn.click(
                            fn=export_agent_handler,
                            inputs=[export_agent_selector],
                            outputs=[export_status, export_file]
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("#### Import Agent")
                        
                        import_file = gr.File(label="Select Agent File to Import")
                        import_overwrite = gr.Checkbox(label="Overwrite if exists", value=False)
                        import_status = gr.Markdown("")
                        import_btn = gr.Button("📥 Import Agent")
                        
                        import_btn.click(
                            fn=import_agent_handler,
                            inputs=[import_file, import_overwrite],
                            outputs=[import_status, agent_selector, chain_agents, edit_agent_selector]
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("#### Backup All Agents")
                        
                        export_all_status = gr.Markdown("")
                        export_all_btn = gr.Button("📦 Export All Agents")
                        export_all_file = gr.File(label="Backup File", interactive=False)
                        
                        export_all_btn.click(
                            fn=export_all_handler,
                            outputs=[export_all_status, export_all_file]
                        )
            
            # ================================================================
            # TAB 5: KNOWLEDGE BASE (RAG)
            # ================================================================
            with gr.Tab("📚 Knowledge Base"):
                gr.Markdown("### RAG (Retrieval Augmented Generation)")
                gr.Markdown("Add documents to enhance agent responses with domain knowledge")
                
                with gr.Tabs():
                    with gr.Tab("📄 Add Document"):
                        doc_title = gr.Textbox(label="Document Title")
                        doc_content = gr.Textbox(label="Content", lines=10)
                        
                        add_doc_status = gr.Markdown("")
                        add_doc_btn = gr.Button("➕ Add Document", variant="primary")
                        
                        add_doc_btn.click(
                            fn=add_knowledge_doc,
                            inputs=[doc_title, doc_content],
                            outputs=[add_doc_status]
                        )
                    
                    with gr.Tab("📊 View Knowledge Base"):
                        refresh_kb_btn = gr.Button("🔄 Refresh")
                        kb_display = gr.Markdown(get_kb_stats())
                        
                        refresh_kb_btn.click(
                            fn=get_kb_stats,
                            outputs=[kb_display]
                        )
                        
                        gr.Markdown("""
                        ### Adding External Documents
                        
                        You can also add documents by placing files in the `knowledge_base/` directory:
                        - `.txt` files for plain text
                        - `.md` files for markdown
                        - `.json` files for structured data
                        
                        Then click Refresh to load them.
                        """)
            
            # ================================================================
            # TAB 6: CANVAS WORKFLOWS
            # ================================================================
            with gr.Tab("🎨 Canvas Workflows"):
                gr.Markdown("### Execute Canvas Workflows")
                gr.Markdown("Import and execute workflows from canvas-based workflow designers")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        workflow_json_input = gr.Textbox(
                            label="Workflow JSON",
                            lines=15,
                            placeholder='Paste your canvas workflow JSON here...\n\nExample format:\n{\n  "version": "1.1.0",\n  "format": "website-playground",\n  "name": "Workflow Name",\n  "graph": { "cells": [...] }\n}',
                            info="Paste the JSON exported from your canvas workflow designer"
                        )
                        
                        with gr.Row():
                            parse_workflow_btn = gr.Button("🔍 Parse Workflow", variant="secondary")
                        
                        workflow_info = gr.Markdown(
                            value="**Instructions:**\n1. Paste your canvas workflow JSON above\n2. Click 'Parse Workflow' to validate and view workflow structure\n3. Enter input text and execute the workflow",
                            label="Workflow Info"
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("### Execute Workflow")
                        
                        workflow_input = gr.Textbox(
                            label="Input Text",
                            lines=4,
                            placeholder="Enter your input for the workflow..."
                        )
                        
                        workflow_context = gr.Textbox(
                            label="Context (Optional)",
                            lines=2,
                            placeholder="Additional context..."
                        )
                        
                        workflow_mode = gr.Radio(
                            choices=["cumulative", "sequential", "parallel"],
                            value="cumulative",
                            label="Pass Mode",
                            info="cumulative: All previous outputs | sequential: Only previous output | parallel: No chaining"
                        )
                        
                        execute_workflow_btn = gr.Button("⚡ Execute Workflow", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Quick Guide")
                        gr.Markdown("""
                        **Features:**
                        - Import workflows from canvas designers
                        - Auto-create agents from workflow config
                        - Visual step-by-step execution
                        - Full workflow history tracking
                        
                        **Workflow Format:**
                        - Supports canvas JSON exports
                        - Agent nodes with config
                        - Trigger-start and end-sink nodes
                        - Standard link connections
                        
                        **Tips:**
                        - Parse workflow first to validate
                        - Agents are created automatically if missing
                        - Check execution order in workflow info
                        """)
                
                with gr.Tabs():
                    with gr.Tab("📊 Visual Output"):
                        workflow_html_output = gr.HTML(
                            value='<div style="text-align:center;padding:40px;color:#9aa4ad;">Parse and execute a workflow to see results...</div>'
                        )
                    
                    with gr.Tab("📝 Raw Text"):
                        workflow_text_output = gr.Textbox(
                            label="Workflow Results",
                            lines=20,
                            placeholder="Workflow execution results will appear here..."
                        )
                
                def show_workflow_loading():
                    loading_html = '''
                    <div style="text-align:center;padding:60px;background:#11161d;border-radius:8px;">
                        <div style="font-size:48px;margin-bottom:20px;">🎨</div>
                        <div style="font-size:18px;color:#8ab4f8;margin-bottom:10px;">Executing Workflow...</div>
                        <div style="font-size:14px;color:#9aa4ad;">Processing your workflow with agents</div>
                        <div style="margin-top:20px;">
                            <div style="width:200px;height:4px;background:#1f2a35;margin:0 auto;border-radius:2px;overflow:hidden;">
                                <div style="width:100%;height:100%;background:linear-gradient(90deg,#8ab4f8,#6a94f8,#8ab4f8);
                                            animation:loading 1.5s ease-in-out infinite;"></div>
                            </div>
                        </div>
                        <style>
                            @keyframes loading {
                                0% { transform: translateX(-100%); }
                                100% { transform: translateX(100%); }
                            }
                        </style>
                    </div>
                    '''
                    return "", loading_html
                
                def parse_workflow_wrapper(json_str: str):
                    """Wrapper to handle parse workflow"""
                    status, _ = parse_workflow_json(json_str)
                    return status
                
                parse_workflow_btn.click(
                    fn=parse_workflow_wrapper,
                    inputs=[workflow_json_input],
                    outputs=[workflow_info]
                )
                
                execute_workflow_btn.click(
                    fn=show_workflow_loading,
                    inputs=None,
                    outputs=[workflow_text_output, workflow_html_output]
                ).then(
                    fn=execute_workflow_handler,
                    inputs=[workflow_json_input, workflow_input, workflow_context, workflow_mode],
                    outputs=[workflow_text_output, workflow_html_output]
                )
            
            # ================================================================
            # TAB 7: HISTORY & STATS
            # ================================================================
            with gr.Tab("📊 History & Statistics"):
                gr.Markdown("### Conversation History and Usage Analytics")
                
                with gr.Tabs():
                    with gr.Tab("📜 History"):
                        refresh_history_btn = gr.Button("🔄 Refresh History")
                        history_display = gr.Markdown(get_history_display())
                        
                        refresh_history_btn.click(
                            fn=get_history_display,
                            outputs=[history_display]
                        )
                    
                    with gr.Tab("📈 Statistics"):
                        refresh_stats_btn = gr.Button("🔄 Refresh Statistics")
                        stats_display = gr.Markdown(get_statistics_display())
                        
                        refresh_stats_btn.click(
                            fn=get_statistics_display,
                            outputs=[stats_display]
                )
        
        gr.Markdown("""
        ---
        ### 💡 Features Overview
        
        - **Run Agents**: Execute multiple agents on any input
        - **Agent Chains**: Sequential execution with data passing
        - **Canvas Workflows**: Import and execute workflows from canvas designers
        - **Templates**: 20+ pre-built agent templates
        - **Edit Agents**: Full editing capabilities for all agents
        - **Import/Export**: Share and backup agents
        - **RAG Integration**: Knowledge-enhanced responses
        - **History Tracking**: Complete conversation logs
        - **Statistics**: Usage analytics and insights
        
        ### 🔒 Safety & Privacy
        
        - All data stored locally
        - Input validation and sanitization
        - Dangerous pattern detection
        - Atomic file operations
        
        **AgentForge Pro v2.0**
        """)
    
    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Initializing AgentForge...")
    initialize(DEFAULT_MODEL)
    
    print("Starting web interface...")
    app = build_interface()
    app.launch(server_name="0.0.0.0", server_port=7870, share=False)

# 🔨 AgentForge Pro - Universal Multi-Agent Manager

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

<img width="2878" height="1642" alt="Screenshot 2025-10-18 201311" src="https://github.com/user-attachments/assets/845e5652-0c60-4ecc-90d4-4871c12915f7" />

Create, manage, and orchestrate specialized AI agents for any purpose. Model-agnostic platform with professional workflows and advanced features.

## Key Features

- **Unlimited Agents**: Create as many specialized agents as needed
- **20+ Professional Templates**: Pre-built expert agents across 7 categories
- **Agent Chaining**: Sequential execution with 3 modes (cumulative, sequential, parallel)
- **RAG Integration**: Knowledge-enhanced responses with document retrieval
- **Multi-Agent Execution**: Run multiple agents simultaneously
- **Full Agent Editing**: Modify agents anytime with built-in editor
- **Conversation History**: Automatic logging of all interactions
- **Usage Analytics**: Track performance and utilization
- **Import/Export**: Share agents and create backups (JSON/ZIP)
- **Model Agnostic**: Compatible with any Hugging Face language model

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure model (optional - edit agent_forge_app.py)
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# 3. Run
python agent_forge_app.py
```

Open `http://localhost:7870` in your browser.

## 📖 Usage Guide

### Creating Agents

**Option 1: Use Templates (Recommended)**
1. Go to **"Create Agent"** tab
2. Select category → Choose template → Click **"Load Template"**
3. Customize if needed → Click **"Create Agent"**

**Option 2: Create from Scratch**
1. Go to **"Create Agent"** tab
2. Fill in Agent Name, Role, and System Prompt
3. Click **"Create Agent"**

### Running Agents

**Standard Execution**
1. Go to **"Run Agents"** tab
2. Enter input text (optional: add context, enable RAG)
3. Select one or more agents → Click **"Run Selected Agents"**

**Agent Chaining**
1. Go to **"Agent Chains"** tab
2. Enter input → Select 2+ agents in order
3. Choose mode: **Cumulative** (all previous outputs), **Sequential** (previous output only), or **Parallel** (comparison)
4. Click **"Run Chain"**

### Managing Agents
- **Edit**: Manage Agents → Edit → Select → Load → Modify → Update
- **Delete**: Manage Agents → Delete → Select → Delete
- **Import/Export**: Manage Agents → Import/Export → Export/Import files

## Configuration

**Models**: Any Hugging Face model (Llama, Mistral, Falcon, etc.)  
**Storage**: Customizable directories for agents, history, and knowledge base  
**Workflows**: Chain agents for complex pipelines (Content Creation, Software QA, Research, etc.)

## Project Structure

```
AgentForge/
├── agent_forge_app.py       # Main Gradio application
├── src/
│   ├── base_agent.py        # Abstract base class
│   ├── custom_agent.py      # Agent implementation
│   ├── agent_manager.py     # CRUD & import/export
│   ├── agent_templates.py   # 20+ pre-built templates
│   ├── agent_chain.py       # Sequential execution
│   ├── history_manager.py   # Conversation logging
│   └── rag_integration.py   # Knowledge base retrieval
├── custom_agents/           # Agent storage (JSON)
├── history/                 # Conversation logs
├── knowledge_base/          # RAG documents
└── requirements.txt         # Dependencies
```

## Safety & Validation

- **Input Validation**: All agent names and prompts validated
- **Dangerous Pattern Detection**: Blocks script tags, eval, exec
- **Length Limits**: Max 100 chars (name), 200 chars (role), 4000 chars (prompt)
- **Local Storage**: All agents stored locally in `custom_agents/`


Future enhancements could include: streaming responses, agent versioning, vector embeddings for advanced RAG, agent collaboration, custom output formatters, REST API mode, and batch processing.

## 📝 License

## Credits

Built on:
- **Gradio**: UI framework
- **LangChain**: LLM orchestration
- **Hugging Face Transformers**: Model inference
- **CyberXP**: Original inspiration for multi-agent system

Citation
If you use this work, please cite:

```bibtex

@software{AgentForge,
  title={🔨 AgentForge Pro - Universal Multi-Agent Manager},
  author={Abaryan},
  year={2025},
  url={https://github.com/abaryan/AgentForge}
}
```

## Acknowledgments

Built with [Gradio](https://gradio.app/), [LangChain](https://www.langchain.com/), and [Hugging Face Transformers](https://huggingface.co/transformers/). Inspired by the multi-agent architecture of [CyberXP](https://github.com/abaryan/CyberXP).

# 🔨 AgentForge Pro - Universal Multi-Agent Manager
<img width="2878" height="1642" alt="Screenshot 2025-10-18 201311" src="https://github.com/user-attachments/assets/845e5652-0c60-4ecc-90d4-4871c12915f7" />


Create, manage, and orchestrate specialized AI agents for any purpose. Model-agnostic platform with professional workflows and advanced features.

## Key Features

### Core Capabilities
- **Create Unlimited Agents**: Build as many specialized agents as you need
- **20+ Professional Templates**: Quick-start with pre-built expert agents
- **Full Agent Editing**: Modify any agent anytime with built-in editor
- **Multi-Agent Execution**: Run multiple agents simultaneously for diverse perspectives
- **Agent Chaining**: Sequential execution with data passing (3 modes)
- **RAG Integration**: Knowledge-enhanced responses with document retrieval
- **Conversation History**: Automatic logging of all agent interactions
- **Usage Analytics**: Track performance and agent utilization
- **Import/Export**: Share agents and create backups
- **Visual Output**: Color-coded responses from each agent
- **Any Domain**: Works for creative writing, code review, data analysis, customer support, etc.
- **Model Agnostic**: Use any Hugging Face language model

###  New in v2.0
- **Agent Templates System** with 20+ professional templates
- **Agent Chaining** with cumulative, sequential, and parallel modes
- **RAG (Retrieval Augmented Generation)** for knowledge-enhanced responses
- **Full Agent Editing** capabilities in management interface
- **Conversation History** with automatic logging
- **Usage Statistics** and analytics dashboard
- **Complete Import/Export** system for agents
- **Enhanced UI** with 6 specialized tabs

- Agent template system with 20+ professional templates
- Agent chaining with 3 execution modes
- RAG (Retrieval Augmented Generation) for knowledge base
- Full agent editing and management interface
- Conversation history with automatic logging
- Usage statistics and analytics dashboard
- Complete import/export system (JSON/ZIP)
- Enhanced UI with smooth loading animations

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
```

# 2. Configure model (optional - edit agent_forge_app.py)
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# 3. Run
python agent_forge_app.py
```

Open `http://localhost:7870` in your browser.

## 📖 Usage Guide

### Creating Agents

#### Option 1: Use Templates (Recommended)
1. Go to **"Create Agent"** tab
2. Select a category (Content Creation, Software Development, etc.)
3. Choose a template from the dropdown
4. Click **"Load Template"**
5. Customize if needed
6. Click **"Create Agent"**

#### Option 2: Create from Scratch
1. Go to **"Create Agent"** tab
2. Fill in:
   - **Agent Name**: Descriptive name (e.g., "Creative Writer", "Code Reviewer")
   - **Role**: Brief description of expertise
   - **System Prompt**: Define behavior, focus areas, output style
3. Click **"Create Agent"**


### Running Agents

#### Standard Execution
1. Go to **"Run Agents"** tab
2. Enter your input text
3. (Optional) Add context for better understanding
4. (Optional) Enable **"Use Knowledge Base (RAG)"** for enhanced responses
5. Select one or more agents from the dropdown
6. Click **"Run Selected Agents"**
7. View results in Visual or Raw Text tabs

#### Agent Chaining  NEW
1. Go to **"Agent Chains"** tab
2. Enter your input text
3. Select 2+ agents in desired order
4. Choose chain mode:
   - **Cumulative**: Each agent sees all previous outputs
   - **Sequential**: Each agent only sees previous output
   - **Parallel**: No chaining (comparison mode)
5. Click **"Run Chain"**
6. View step-by-step results

### Managing Agents
- **Edit**: Manage Agents tab → Edit sub-tab → Select → Load → Modify → Update
- **Delete**: Manage Agents tab → Delete sub-tab → Select → Delete
- **Export/Import**: Manage Agents tab → Import/Export sub-tab → Export single/all or Import from file

1. Go to **"Manage Agents"** tab
2. **Edit**: Modify existing agents
3. **View All**: See complete agent list
4. **Delete**: Remove agents permanently
5. **Import/Export**: Share agents or create backups

## Example Use Cases

All templates are built-in and ready to use.

## Configuration

**Models**: Compatible with any Hugging Face model (Llama, Mistral, Falcon, etc.)  
**Storage**: Customizable directories for agents, history, and knowledge base  
**Workflows**: Chain multiple agents for complex pipelines (Content Creation, Software QA, Research, etc.)

## Project Structure

```
AgentForge/
├── agent_forge_app.py       # Main Gradio application (v2.0)
├── src/
│   ├── base_agent.py        # Abstract base class
│   ├── custom_agent.py      # Agent implementation
│   ├── agent_manager.py     # CRUD operations + import/export
│   ├── agent_templates.py   # 20+ pre-built templates
│   ├── agent_chain.py       # Sequential execution
│   ├── history_manager.py   # Conversation logging
│   └── rag_integration.py   # Knowledge base retrieval
├── custom_agents/           # Agent storage (JSON)
├── history/                 # Conversation logs
├── knowledge_base/          # RAG documents
└── requirements.txt         # Dependencies
```

##  Safety & Validation

- **Input Validation**: All agent names and prompts are validated
- **Dangerous Pattern Detection**: Blocks script tags, eval, exec
- **Length Limits**: Max 100 chars (name), 200 chars (role), 4000 chars (prompt)
- **Local Storage**: All agents stored locally in `custom_agents/`

## 🆕 What's New in v2.0

### Major Features Added
-  **Agent Templates**: 20+ professional templates across 7 categories
-  **Agent Chaining**: Sequential execution with 3 modes
-  **RAG Integration**: Knowledge-enhanced responses
-  **Full Agent Editing**: Modify agents anytime
-  **Conversation History**: Automatic logging system
-  **Usage Analytics**: Performance metrics dashboard
-  **Import/Export**: Complete agent sharing system
-  **Enhanced UI**: 6 specialized tabs

### Improvements
- Better error handling and validation
- Atomic file operations for data safety
- Color-coded visual outputs
- Responsive UI with better organization
- Comprehensive documentation


## 🤝 Contributing

AgentForge is designed to be extended. Future enhancements could include:
-  Streaming responses (real-time output)
-  Agent versioning system
-  Vector embeddings for advanced RAG
-  Agent collaboration (agents calling agents)
-  Custom output formatters (JSON, Markdown, HTML)
-  REST API mode
-  Batch processing capabilities

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


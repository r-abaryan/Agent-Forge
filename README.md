# 🔨 AgentForge Pro - Universal Multi-Agent Manager
<img width="2878" height="1642" alt="Screenshot 2025-10-18 201311" src="https://github.com/user-attachments/assets/845e5652-0c60-4ecc-90d4-4871c12915f7" />


Create, manage, and orchestrate multiple AI agents for **any purpose** with advanced features and professional workflows.

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

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Model

Edit `agent_forge_app.py` and change the model:

```python
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # Change to your preferred model
```

### 3. Run the App

```bash
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

### Editing Agents  NEW

1. Go to **"Manage Agents"** tab → **"Edit Agent"** sub-tab
2. Select agent from dropdown
3. Click **"Load Agent"**
4. Modify name, role, or system prompt
5. Click **"Update Agent"**

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

1. Go to **"Manage Agents"** tab
2. **Edit**: Modify existing agents
3. **View All**: See complete agent list
4. **Delete**: Remove agents permanently
5. **Import/Export**: Share agents or create backups

## Example Use Cases

### Creative Writing
Create agents for:
- Short Story Writer
- Poet
- Screenplay Writer
- Blog Post Writer

### Software Development
Create agents for:
- Code Reviewer
- Bug Finder
- Documentation Writer
- Refactoring Expert

### Business
Create agents for:
- Marketing Copywriter
- Product Analyst
- Customer Support
- Email Responder

### Education
Create agents for:
- Tutor (Math)
- Tutor (Science)
- Essay Grader
- Study Guide Creator

##  Agent Examples

### Example 1: Creative Writer

```
Name: Creative Writer
Role: Expert in creative storytelling
Prompt:
You are an expert creative writer specializing in short stories.

Focus on:
- Compelling character development
- Unexpected plot twists
- Vivid, sensory descriptions
- Emotional resonance

Provide engaging, well-structured stories with a clear beginning, middle, and end.
```

### Example 2: Code Reviewer

```
Name: Code Reviewer
Role: Expert in code quality and best practices
Prompt:
You are a senior software engineer reviewing code.

Focus on:
- Code quality and readability
- Security vulnerabilities
- Performance issues
- Best practices and patterns

Provide constructive feedback with specific suggestions for improvement.
```

### Example 3: Data Analyst

```
Name: Data Analyst
Role: Expert in data interpretation and insights
Prompt:
You are a data analyst specializing in business intelligence.

Focus on:
- Identifying patterns and trends
- Providing actionable insights
- Clear, data-driven recommendations
- Visual descriptions (when applicable)

Analyze data thoroughly and provide clear, business-focused conclusions.
```

## 🔧 Advanced Configuration

### Using Different Models

AgentForge works with any Hugging Face model. Popular choices:

- `meta-llama/Llama-2-7b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `tiiuae/falcon-7b-instruct`
- `google/flan-t5-xl`

### Custom Storage Location

Change the storage directory in `agent_forge_app.py`:

```python
agent_manager = AgentManager(storage_dir="my_custom_agents")
```

### Multi-Agent Workflows

Combine agents for comprehensive analysis:
- **Content Creation**: Idea Generator + Writer + Editor
- **Software QA**: Bug Finder + Security Auditor + Performance Analyst
- **Research**: Data Gatherer + Analyst + Report Writer

## 📁 Project Structure

```
AgentForge/
├── agent_forge_app.py       # Main Gradio application (v2.0)
├── src/
│   ├── base_agent.py        # Abstract base class
│   ├── custom_agent.py      # Custom agent implementation
│   ├── agent_manager.py     # Agent storage/retrieval + import/export
│   ├── agent_templates.py   # 20+ pre-built templates  NEW
│   ├── agent_chain.py       # Sequential execution system  NEW
│   ├── history_manager.py   # Conversation logging  NEW
│   └── rag_integration.py   # Knowledge base retrieval  NEW
├── custom_agents/           # Saved agents (JSON)
├── history/                 # Conversation logs  NEW
├── knowledge_base/          # RAG documents  NEW
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── FEATURES.md             # Complete feature guide  NEW
├── QUICK_START.md          # Getting started guide
└── PROJECT_SUMMARY.md      # Project overview
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

@software{AgentForge,
  title={🔨 AgentForge Pro - Universal Multi-Agent Manager},
  author={Abaryan},
  year={2025}
}
---


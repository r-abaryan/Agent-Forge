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
 - **Proper Logging**: Detailed logging per run

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

## Usage (Short)

- **Create Agents**: Use templates or define from scratch in the **Create Agent** tab.
- **Run Agents**: Use the **Run Agents** tab for single or multi-agent runs (optionally with RAG).
- **Chains & Workflows**: Use **Agent Chains** or **Canvas Workflows** for multi-step pipelines.
- **Manage Agents**: Edit, delete, import, and export agents from the management tabs.


## Safety & Validation

- **Input Validation** on agent names, roles, and prompts
- **Dangerous Pattern Detection** for script/eval/exec
- **Length Limits** on all user-provided fields
- **Local Storage Only** for agents and history

## Acknowledgments

Built with [Gradio](https://gradio.app/), [LangChain](https://www.langchain.com/), and [Hugging Face Transformers](https://huggingface.co/transformers/).



## Citation

If you use this work, please cite:

```bibtex
@software{CyberXP,
  title={CyberXP: AI-Powered Cyber Threat Assessment with Multi-Agent Architecture},
  author={Abaryan},
  year={2025},
  url={https://github.com/r-abaryan/CyberLLM-Agent}
}
```
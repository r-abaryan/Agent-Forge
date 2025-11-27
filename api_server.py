"""
AgentForge REST API Server
FastAPI server for programmatic access to agents
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

from src.custom_agent import CustomAgent
from src.agent_manager import AgentManager
from src.agent_chain import AgentChain
from src.rag_integration import SimpleRAG
from src.api_models import (
    AgentExecuteRequest, AgentExecuteResponse,
    AgentChainRequest, AgentChainResponse,
    AgentCreateRequest, AgentUpdateRequest,
    AgentInfo, AgentListResponse,
    DocumentAddRequest, DocumentSearchRequest, DocumentSearchResponse,
    ErrorResponse, SuccessResponse
)
from src.api_auth import verify_api_key, api_key_manager


# Configuration
DEFAULT_MODEL = "abaryan/CyberXP_Agent_Llama_3.2_1B"

# Global instances
llm = None
agent_manager = None
agent_chain = None
rag_system = None


def initialize_system(model_path: str = DEFAULT_MODEL):
    """Initialize LLM and all manager instances"""
    global llm, agent_manager, agent_chain, rag_system
    
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
    
    # Initialize managers
    agent_manager = AgentManager(storage_dir="custom_agents")
    agent_chain = AgentChain(agent_manager, llm)
    rag_system = SimpleRAG(knowledge_base_dir="knowledge_base")
    
    print("System initialized successfully!")


# Create FastAPI app
app = FastAPI(
    title="AgentForge API",
    description="REST API for AgentForge multi-agent system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    initialize_system()


# ============================================================================
# AGENT EXECUTION ENDPOINTS
# ============================================================================

@app.post("/api/v1/agents/execute", response_model=AgentExecuteResponse)
async def execute_agent(
    request: AgentExecuteRequest,
    api_key: str = Depends(verify_api_key)
):
    """Execute a single agent"""
    try:
        # Load agent
        agent = agent_manager.load_agent(request.agent_name, llm=llm)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{request.agent_name}' not found")
        
        # Add RAG context if requested
        context = request.context
        if request.use_rag:
            rag_context = rag_system.get_context_for_query(request.input_text, top_k=3)
            context = f"{context}\n\n{rag_context}" if context else rag_context
        
        # Execute agent
        result = agent.process(request.input_text, context=context)
        
        return AgentExecuteResponse(
            success=result.get("success", False),
            agent=result.get("agent", request.agent_name),
            role=result.get("role", ""),
            response=result.get("response", ""),
            error=None if result.get("success") else result.get("response")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/agents/execute/stream")
async def execute_agent_stream(
    request: AgentExecuteRequest,
    api_key: str = Depends(verify_api_key)
):
    """Execute agent with streaming response"""
    try:
        # Load agent
        agent = agent_manager.load_agent(request.agent_name, llm=llm)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{request.agent_name}' not found")
        
        # Add RAG context if requested
        context = request.context
        if request.use_rag:
            rag_context = rag_system.get_context_for_query(request.input_text, top_k=3)
            context = f"{context}\n\n{rag_context}" if context else rag_context
        
        # Stream response
        def generate():
            for chunk in agent.process_stream(request.input_text, context=context):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/agents/chain", response_model=AgentChainResponse)
async def execute_agent_chain(
    request: AgentChainRequest,
    api_key: str = Depends(verify_api_key)
):
    """Execute a chain of agents"""
    try:
        result = agent_chain.execute_chain(
            agent_names=request.agent_names,
            initial_input=request.input_text,
            context=request.context,
            pass_mode=request.pass_mode
        )
        
        return AgentChainResponse(
            success=result.get("success", True),
            workflow_name=f"Chain: {' -> '.join(request.agent_names)}",
            total_steps=result.get("total_steps", 0),
            successful_steps=result.get("successful_steps", 0),
            results=result.get("results", []),
            final_output=result.get("final_output", ""),
            error=None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AGENT MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/v1/agents", response_model=AgentListResponse)
async def list_agents(api_key: str = Depends(verify_api_key)):
    """List all agents"""
    agents = agent_manager.list_agents()
    return AgentListResponse(agents=agents, total=len(agents))


@app.get("/api/v1/agents/{agent_name}", response_model=AgentInfo)
async def get_agent(agent_name: str, api_key: str = Depends(verify_api_key)):
    """Get agent details"""
    agent_data = agent_manager.get_agent_data(agent_name)
    if not agent_data:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    return AgentInfo(**agent_data)


@app.post("/api/v1/agents", response_model=SuccessResponse)
async def create_agent(
    request: AgentCreateRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a new agent"""
    try:
        if agent_manager.agent_exists(request.name):
            raise HTTPException(status_code=400, detail=f"Agent '{request.name}' already exists")
        
        agent = CustomAgent(
            name=request.name,
            role=request.role,
            system_prompt=request.system_prompt,
            llm=llm,
            few_shot_examples=request.few_shot_examples
        )
        
        success = agent_manager.save_agent(agent)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save agent")
        
        return SuccessResponse(
            success=True,
            message=f"Agent '{request.name}' created successfully",
            data={"name": request.name}
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/agents/{agent_name}", response_model=SuccessResponse)
async def update_agent(
    agent_name: str,
    request: AgentUpdateRequest,
    api_key: str = Depends(verify_api_key)
):
    """Update an existing agent"""
    try:
        # Get existing agent
        agent_data = agent_manager.get_agent_data(agent_name)
        if not agent_data:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Update fields
        new_name = request.name if request.name else agent_data["name"]
        new_role = request.role if request.role else agent_data["role"]
        new_prompt = request.system_prompt if request.system_prompt else agent_data["system_prompt"]
        new_examples = request.few_shot_examples if request.few_shot_examples is not None else agent_data.get("few_shot_examples", "")
        
        # Create updated agent
        agent = CustomAgent(
            name=new_name,
            role=new_role,
            system_prompt=new_prompt,
            llm=llm,
            few_shot_examples=new_examples
        )
        
        success = agent_manager.update_agent(agent_name, agent)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update agent")
        
        return SuccessResponse(
            success=True,
            message=f"Agent '{new_name}' updated successfully",
            data={"name": new_name}
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/agents/{agent_name}", response_model=SuccessResponse)
async def delete_agent(agent_name: str, api_key: str = Depends(verify_api_key)):
    """Delete an agent"""
    success = agent_manager.delete_agent(agent_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    return SuccessResponse(
        success=True,
        message=f"Agent '{agent_name}' deleted successfully"
    )


# ============================================================================
# KNOWLEDGE BASE ENDPOINTS
# ============================================================================

@app.post("/api/v1/knowledge/documents", response_model=SuccessResponse)
async def add_document(
    request: DocumentAddRequest,
    api_key: str = Depends(verify_api_key)
):
    """Add a document to the knowledge base"""
    success = rag_system.add_document(
        title=request.title,
        content=request.content,
        metadata=request.metadata
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add document")
    
    return SuccessResponse(
        success=True,
        message=f"Document '{request.title}' added successfully"
    )


@app.post("/api/v1/knowledge/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    api_key: str = Depends(verify_api_key)
):
    """Search knowledge base"""
    results = rag_system.search(request.query, top_k=request.top_k)
    
    return DocumentSearchResponse(
        results=results,
        total=len(results)
    )


@app.get("/api/v1/knowledge/documents", response_model=DocumentSearchResponse)
async def list_documents(api_key: str = Depends(verify_api_key)):
    """List all documents in knowledge base"""
    documents = rag_system.list_documents()
    
    return DocumentSearchResponse(
        results=documents,
        total=len(documents)
    )


# ============================================================================
# API KEY MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/api/v1/admin/keys/generate", response_model=SuccessResponse)
async def generate_api_key(
    name: str,
    description: str = "",
    api_key: str = Depends(verify_api_key)
):
    """Generate a new API key (admin only)"""
    new_key = api_key_manager.generate_key(name, description)
    
    return SuccessResponse(
        success=True,
        message="API key generated successfully",
        data={"api_key": new_key, "name": name}
    )


@app.get("/api/v1/admin/keys")
async def list_api_keys(api_key: str = Depends(verify_api_key)):
    """List all API keys (without showing actual keys)"""
    keys = api_key_manager.list_keys()
    return {"keys": keys, "total": len(keys)}


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "agents_available": len(agent_manager.list_agents()) if agent_manager else 0
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AgentForge API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

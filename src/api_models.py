"""
API Models - Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class AgentExecuteRequest(BaseModel):
    """Request model for agent execution"""
    agent_name: str = Field(..., description="Name of the agent to execute")
    input_text: str = Field(..., description="Input text for the agent")
    context: Optional[str] = Field("", description="Additional context")
    use_rag: bool = Field(False, description="Whether to use RAG for context retrieval")


class AgentExecuteResponse(BaseModel):
    """Response model for agent execution"""
    success: bool
    agent: str
    role: str
    response: str
    error: Optional[str] = None


class AgentChainRequest(BaseModel):
    """Request model for agent chain execution"""
    agent_names: List[str] = Field(..., description="List of agent names in execution order")
    input_text: str = Field(..., description="Initial input text")
    context: Optional[str] = Field("", description="Additional context")
    pass_mode: str = Field("cumulative", description="Data passing mode: cumulative, sequential, or parallel")


class AgentChainResponse(BaseModel):
    """Response model for agent chain execution"""
    success: bool
    workflow_name: str
    total_steps: int
    successful_steps: int
    results: List[Dict[str, Any]]
    final_output: str
    error: Optional[str] = None


class AgentCreateRequest(BaseModel):
    """Request model for creating an agent"""
    name: str = Field(..., max_length=100, description="Agent name")
    role: str = Field(..., max_length=200, description="Agent role/specialty")
    system_prompt: str = Field(..., max_length=4000, description="System prompt defining behavior")
    few_shot_examples: Optional[str] = Field("", description="Optional few-shot examples")


class AgentUpdateRequest(BaseModel):
    """Request model for updating an agent"""
    name: Optional[str] = Field(None, max_length=100, description="New agent name")
    role: Optional[str] = Field(None, max_length=200, description="New agent role")
    system_prompt: Optional[str] = Field(None, max_length=4000, description="New system prompt")
    few_shot_examples: Optional[str] = Field(None, description="New few-shot examples")


class AgentInfo(BaseModel):
    """Agent information model"""
    name: str
    role: str
    system_prompt: str
    few_shot_examples: str = ""


class AgentListResponse(BaseModel):
    """Response model for listing agents"""
    agents: List[Dict[str, str]]
    total: int


class DocumentAddRequest(BaseModel):
    """Request model for adding a document to knowledge base"""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class DocumentSearchRequest(BaseModel):
    """Request model for searching documents"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(3, ge=1, le=20, description="Number of results to return")


class DocumentSearchResponse(BaseModel):
    """Response model for document search"""
    results: List[Dict[str, Any]]
    total: int


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None


class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

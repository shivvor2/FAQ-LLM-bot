"""Pydantic models for HTTP endpoints validation and documentation"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, List, Union, Literal
from enum import Enum


# Assistant initialization
class ClientArgs(BaseModel):
    """Configuration arguments for initializing the OpenAI client."""

    api_key: str = Field(
        None,
        description="API key for authentication for the inference provider, If not provided, defaults to the `OPENAI_API_KEY` environmental variable in the host environment. Leave blank unless there is a reason to override this setting",
    )
    base_url: Optional[str] = Field(
        None,
        description="Base URL for API requests. If not provided, defaults to the `OPENAI_BASE_URL` environmental variable in the host environment. Leave blank unless there is a reason to override this setting",
        example="https://openrouter.ai/api/v1/",
        alias="baseUrl",
    )
    timeout: Optional[float] = Field(
        None,
        description="Timeout in seconds for API requests",
        gt=0,
        example=30.0,
    )
    max_retries: Optional[int] = Field(
        None,
        description="Maximum number of retries for API calls",
        ge=0,
        example=2,
        alias="maxRetries",
    )

    class Config:
        populate_by_name = True


class QnAArgs(BaseModel):
    """Configuration arguments for the QnA assistant's behavior."""

    prompt: str = Field(
        None,
        description="System prompt to guide assistant behavior, if not provided, will default to a default prompt hosted in the deployment environment",
        example="You are a helpful assistant specialized in answering questions about our FAQ...",
    )
    prompt_name: str = Field(
        None,
        description='Name of prompt (in the prompts folder, specified by env variable `PROMPTS_PATH`), for example, if I set this value to be "prompt1", then the endpoint will attempt to load "prompt1.txt" in the prompts folder. When both `prompt` and `promptName` is provided, `prompt` will take presidence',
        alias="promptName",
    )
    model: str = Field(
        None,
        description="OpenAI model to use for generating responses, If not provided, defaults to the `OPENAI_MODEL` environmental variable in the host environment. Leave blank unless there is a reason to override this setting",
        example="nousresearch/hermes-3-llama-3.1-405b",
    )
    temperature: Optional[float] = Field(
        None,
        description="Temperature for response generation (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.7,
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum number of tokens in assistant's response",
        gt=0,
        example=1000,
        alias="maxTokens",
    )
    top_p: Optional[float] = Field(
        None,
        description="Top-p sampling parameter",
        ge=0.0,
        le=1.0,
        example=1.0,
        alias="topP",
    )
    frequency_penalty: Optional[float] = Field(
        None,
        description="Frequency penalty parameter",
        ge=-2.0,
        le=2.0,
        example=0.0,
        alias="frequencyPenalty",
    )
    presence_penalty: Optional[float] = Field(
        None,
        description="Presence penalty parameter",
        ge=-2.0,
        le=2.0,
        example=0.0,
        alias="presencePenalty",
    )
    additionals: Optional[Dict[str, str]] = Field(
        None,
        description="Additional information to initialize the assistant with",
    )
    chat_history: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Previous conversation history to initialize the assistant with",
        alias="chatHistory",
    )

    class Config:
        populate_by_name = True


class InitParams(BaseModel):
    """Parameters for initializing a new QnA session."""

    client_args: ClientArgs = Field(
        ...,
        description="Arguments for constructing the OpenAI client",
        alias="clientArgs",
    )
    qna_args: QnAArgs = Field(
        ...,
        description="Arguments for constructing the QnA instance",
        alias="qnaArgs",
    )

    class Config:
        populate_by_name = True


# Messaging
class MessageRequest(BaseModel):
    """Request model for sending messages to the assistant."""

    message: str = Field(
        ...,
        description="The message text to be processed by the assistant",
        example="What does your FAQ say about return policies?",
    )
    stream: Optional[bool] = Field(
        False,
        description="Whether to stream the response",
    )

    class Config:
        populate_by_name = True


class StreamRequest(BaseModel):
    """Request model for creating a streaming job."""

    message: str = Field(
        ...,
        description="The message text to be processed by the assistant",
        example="What does your FAQ say about return policies?",
    )

    class Config:
        populate_by_name = True


# Additional information
class RestCallInfo(BaseModel):
    """Information for making a REST API call."""

    base_url: HttpUrl = Field(
        ...,
        description="Base URL for the REST call",
        example="https://api.example.com/v1",
        alias="baseUrl",
    )
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = Field(
        ...,
        description="HTTP method for the REST call",
        example="GET",
    )
    headers: Optional[Dict[str, str]] = Field(
        None,
        description="Headers for the REST call",
        example={"Authorization": "Bearer token123"},
    )
    params: Optional[Dict[str, Any]] = Field(
        None,
        description="Parameters for the REST call (query params for GET, body for POST/PUT)",
        example={"limit": 10, "offset": 0},
    )

    class Config:
        populate_by_name = True


class AdditionalRequestItem(BaseModel):
    """A single request for adding additional information."""

    id: str = Field(
        ...,
        description="Unique identifier for this additional information. Each Additional information can be removed from the context using the Identifier",
        example="pricing_data",
    )
    description: Optional[str] = Field(
        None,
        description="Description of what this additional information represents",
        example="Current pricing data for our premium plan",
    )
    content: Union[str, RestCallInfo] = Field(
        ...,
        description="The content to add - either a direct string or REST call information",
    )

    class Config:
        populate_by_name = True


class AdditionalsRequest(BaseModel):
    """Request model for adding additional information to a session."""

    items: List[AdditionalRequestItem] = Field(
        ...,
        description="List of additional information items to add",
        min_items=1,
    )

    class Config:
        populate_by_name = True


class RemoveAdditionalsRequest(BaseModel):
    """Request model for removing additional information from a session."""

    ids: List[str] = Field(
        ...,
        description="List of additional information IDs to remove",
        min_items=1,
        example=["pricing_data", "user_stats"],
    )

    class Config:
        populate_by_name = True


# This is for the `get_session_state` endpoint, is currently not used, but will be used when we refactor the code to use a custom data class for response of each endpoint
class ChatHistoryItem(BaseModel):
    """Model for a single chat history message item."""

    role: str = Field(
        ...,
        description="Role of the message sender (user, assistant, or system)",
        example="user",
    )
    content: Union[str, List] = Field(
        ...,
        description="Content of the message",
        example="What does your FAQ say about return policies?",
    )


class SessionStateResponse(BaseModel):
    """Response model with session state information."""

    chat_history: List[ChatHistoryItem] = Field(
        ...,
        description="List of messages in the conversation history",
        alias="chatHistory",
    )
    additionals: Dict[str, str] = Field(
        ...,
        description="Additional information/context currently associated with the session",
        example={
            "policy_data": "Our return policy allows returns within 30 days of purchase..."
        },
    )

    class Config:
        populate_by_name = True


# Response
class MetaData(BaseModel):
    """Metadata included in all service responses."""

    message_id: str = Field(
        ...,
        description="Unique identifier for the response",
        example="123e4567-e89b-12d3-a456-426614174000",
        alias="messageID",
    )
    timestamp: str = Field(
        ...,
        description="ISO format timestamp of when the response was generated",
        example="2024-01-20T12:34:56.789Z",
    )

    class Config:
        populate_by_name = True


class ServiceResponse(BaseModel):
    """Standard response model for all service endpoints."""

    payload: Optional[Dict[str, Any]] = Field(
        None, description="Response payload containing endpoint-specific data"
    )
    metadata: MetaData = Field(
        ..., description="Response metadata including message ID and timestamp"
    )

    class Config:
        populate_by_name = True


class StreamJobResponse(BaseModel):
    """Response model for stream job creation."""

    job_id: str = Field(
        ...,
        description="Unique identifier for the streaming job",
        example="job-123e4567-e89b-12d3-a456-426614174000",
        alias="jobId",
    )

    class Config:
        populate_by_name = True


# Health
class HealthStatus(str, Enum):
    """Possible health status values for the service."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthMetrics(BaseModel):
    """Metrics about the service's current state."""

    active_sessions: int = Field(
        ...,
        description="Total number of existing QnA sessions",
        example=5,
        ge=0,
        alias="activeSessions",
    )
    locked_sessions: int = Field(
        ...,
        description="Number of sessions currently processing messages",
        example=1,
        ge=0,
        alias="lockedSessions",
    )

    class Config:
        populate_by_name = True


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: HealthStatus = Field(
        ..., description="Current health status of the service"
    )
    error: Optional[str] = Field(
        None, description="Error message if service is unhealthy"
    )
    version: str = Field(..., description="Current API version", example="1.0.0")
    timestamp: str = Field(
        ...,
        description="ISO format timestamp of the health check",
        example="2024-01-20T12:34:56.789Z",
    )
    metrics: Optional[HealthMetrics] = Field(
        None,
        description="Current service metrics (only returned if check logic runs successfully)",
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "error": None,
                "version": "1.0.0",
                "timestamp": "2024-01-20T12:34:56.789Z",
                "metrics": {"activeSessions": 5, "lockedSessions": 1},
            }
        }

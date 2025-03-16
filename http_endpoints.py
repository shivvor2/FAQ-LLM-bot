"""FastAPI endpoints for the QnA service."""

from fastapi import FastAPI, HTTPException, Response, Header, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Tuple
from asyncio import Lock
import uuid
import os
from datetime import datetime
import logging
from openai import OpenAI
from fastapi.responses import StreamingResponse

# Import from our modules
from document_qna.qna import QnA
from http_endpoints_types import (
    InitParams,
    MessageRequest,
    StreamRequest,
    AdditionalsRequest,
    RemoveAdditionalsRequest,
    MetaData,
    ServiceResponse,
    HealthStatus,
    HealthMetrics,
    HealthResponse,
)
from rest_helper import make_rest_call

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Create versioned router
mount_prefix = os.getenv("MOUNT_PREFIX", "")
real_prefix = "/" + mount_prefix if mount_prefix else ""
v1_router = APIRouter(prefix="/v1")


# Default prompt
# (Current prompt is a placeholder: Shv2 1-3-25)
with open("prompt/default_prompt.txt", "r") as file:
    default_prompt = file.read()

# Store QnA instances and their locks
qna_instances: Dict[str, Tuple[QnA, Lock]] = {}

# Store streaming jobs
streaming_jobs: Dict[str, Tuple[str, str]] = {}  # job_id -> (session_id, message)


def get_metadata() -> MetaData:
    """Generate metadata for a response."""
    return MetaData.model_validate(
        {"messageID": str(uuid.uuid4()), "timestamp": datetime.now().isoformat()}
    )


def get_prompt_from_name(name: str):
    "Gets prompt by prompt name"
    try:
        prompts_folder_path = os.getenv("PROMPTS_PATH", "")
        prompt_path = os.path.join(prompts_folder_path, name + ".txt")
        with open(prompt_path, "r") as file:
            prompt = file.read()
            return prompt
    except FileNotFoundError:
        logger.warning(f"Prompt: {name}.txt not found")
        return None


@v1_router.post("/init", tags=["session"], response_model=ServiceResponse)
async def initialize_qna(init_params: InitParams):
    """
    Initialize a new QnA instance with specified configuration.

    Creates a new session with a unique ID and initializes a QnA instance
    with the provided configuration.

    Parameters:
        init_params: Configuration parameters including:
            - clientArgs: Required OpenAI client configuration
            - qnaArgs: QnA instance settings (prompt, model, etc.)

    Returns:
        ServiceResponse containing:
            - payload: Dictionary with the new session ID
            - metadata: Request metadata including message ID and timestamp

    Raises:
        HTTPException(500): For any initialization errors
    """
    session_id = str(uuid.uuid4())

    while session_id in qna_instances:
        logger.warning("Session ID collision occurred, generating new ID")
        session_id = str(uuid.uuid4())

    try:
        # Extract client arguments
        client_args = {
            "api_key": init_params.client_args.api_key or os.getenv("OPENAI_API_KEY"),
            "base_url": init_params.client_args.base_url
            or os.getenv("OPENAI_BASE_URL"),
            "timeout": init_params.client_args.timeout,
            "max_retries": init_params.client_args.max_retries,
        }
        client_args = {k: v for k, v in client_args.items() if v is not None}

        # Create OpenAI client
        client = OpenAI(**client_args)

        # Extract QnA arguments
        qna_args = {
            "client": client,
            "prompt": init_params.qna_args.prompt
            or get_prompt_from_name(init_params.qna_args.prompt_name)
            or default_prompt
            or "",
            "additionals": init_params.qna_args.additionals,
            "chat_history": init_params.qna_args.chat_history,
        }

        # Extract client_args for QnA (model parameters)
        client_config = {
            "model": init_params.qna_args.model or os.getenv("OPENAI_MODEL"),
            "temperature": init_params.qna_args.temperature,
            "max_tokens": init_params.qna_args.max_tokens,
            "top_p": init_params.qna_args.top_p,
            "frequency_penalty": init_params.qna_args.frequency_penalty,
            "presence_penalty": init_params.qna_args.presence_penalty,
        }
        client_config = {k: v for k, v in client_config.items() if v is not None}

        qna_args["client_args"] = client_config

        # Create QnA instance
        qna_instance = QnA(**qna_args)

        # Store instance with lock
        qna_instances[session_id] = (qna_instance, Lock())

        payload = {"sessionId": session_id}
        return ServiceResponse(payload=payload, metadata=get_metadata())

    except Exception as e:
        logger.error(f"Error in initialization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.post("/message", tags=["conversation"], response_model=ServiceResponse)
async def process_message(
    message_request: MessageRequest,
    x_session_id: str = Header(
        ...,
        alias="X-Session-ID",
        description="Active session ID obtained from /init endpoint",
    ),
):
    """
    Process a message using the QnA instance within a specific session.

    Responses are returned statically, for streaming responses, please use the "/stream" endpoint

    This endpoint processes a single message in the context of the specified session.
    Only one message can be processed at a time per session (concurrent requests are prevented).

    Parameters:
        message_request: The message to be processed
        x_session_id: Session ID provided in X-Session-ID header

    Returns:
        ServiceResponse containing:
            - payload: Dictionary with the assistant's response message
            - metadata: Request metadata including message ID and timestamp

    Raises:
        HTTPException(404): If the specified session is not found
        HTTPException(409): If the session is currently processing another message
        HTTPException(500): For any processing errors
    """
    if x_session_id not in qna_instances:
        raise HTTPException(
            status_code=404, detail=f"Session {x_session_id} not initialized"
        )

    if not message_request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    qna_instance, lock = qna_instances[x_session_id]

    if lock.locked():
        raise HTTPException(
            status_code=409, detail="Session is currently processing another message"
        )

    async with lock:
        try:
            # If streaming is requested, create a streaming job instead
            if message_request.stream:
                job_id = f"job-{str(uuid.uuid4())}"
                streaming_jobs[job_id] = (x_session_id, message_request.message)

                payload = {"jobId": job_id}
                return ServiceResponse(payload=payload, metadata=get_metadata())

            # Otherwise, process message normally
            response = qna_instance(message_request.message)
            payload = {"message": response}
            return ServiceResponse(payload=payload, metadata=get_metadata())
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@v1_router.post("/stream", tags=["conversation"], response_model=ServiceResponse)
async def create_stream_job(
    stream_request: StreamRequest,
    x_session_id: str = Header(
        ...,
        alias="X-Session-ID",
        description="Active session ID obtained from /init endpoint",
    ),
):
    """
    Create a streaming job for processing a message.

    This endpoint creates a job for streaming a response to a message.
    The job ID can be used with the /stream/{job_id} endpoint to get the streaming response.

    Parameters:
        stream_request: The message to be processed
        x_session_id: Session ID provided in X-Session-ID header

    Returns:
        ServiceResponse containing:
            - payload: Dictionary with the job ID
            - metadata: Request metadata including message ID and timestamp

    Raises:
        HTTPException(404): If the specified session is not found
        HTTPException(409): If the session is currently processing another message
    """
    if x_session_id not in qna_instances:
        raise HTTPException(
            status_code=404, detail=f"Session {x_session_id} not initialized"
        )

    if not stream_request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    qna_instance, lock = qna_instances[x_session_id]

    if lock.locked():
        raise HTTPException(
            status_code=409, detail="Session is currently processing another message"
        )

    # Create a streaming job
    job_id = f"job-{str(uuid.uuid4())}"
    streaming_jobs[job_id] = (x_session_id, stream_request.message)

    payload = {"jobId": job_id}
    return ServiceResponse(payload=payload, metadata=get_metadata())


@v1_router.get("/stream/{job_id}", tags=["conversation"])
async def get_stream_response(job_id: str):
    """
    Get a streaming response for a previously created streaming job.

    Parameters:
        job_id: ID of the streaming job created with the /stream endpoint

    Returns:
        StreamingResponse: A streaming response with the assistant's message

    Raises:
        HTTPException(404): If the specified job ID is not found
        HTTPException(409): If the session is currently processing another message
    """
    if job_id not in streaming_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    session_id, message = streaming_jobs.pop(job_id)

    if session_id not in qna_instances:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    qna_instance, lock = qna_instances[session_id]

    if lock.locked():
        raise HTTPException(
            status_code=409, detail="Session is currently processing another message"
        )

    async def stream_generator():
        async with lock:
            try:
                for chunk in qna_instance(message, stream=True):
                    yield chunk
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}", exc_info=True)
                yield f"Error: {str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")


@v1_router.get("/state", tags=["conversation"], response_model=ServiceResponse)
async def get_session_state(
    x_session_id: str = Header(
        ...,
        alias="X-Session-ID",
        description="Active session ID for which to retrieve state information",
    ),
):
    """
    Retrieve the current state of a QnA session, including chat history and additional context.

    This endpoint returns the raw chat history (all messages exchanged between the user and assistant)
    and any additional information/context that has been added to the session.

    The chat history consists of a list of message items, where each item contains:
    - role: String indicating the sender ('user', 'assistant', or 'system')
    - content: Text content of the message

    Parameters:
        x_session_id: Session ID provided in X-Session-ID header

    Returns:
        ServiceResponse containing:
            - payload: Dictionary with session state information
                - chatHistory: List of messages in chronological order
                - additionals: Dictionary of additional context information
            - metadata: Request metadata including message ID and timestamp

    Raises:
        HTTPException(404): If the specified session is not found
        HTTPException(500): For any errors retrieving session state
    """
    if x_session_id not in qna_instances:
        raise HTTPException(status_code=404, detail=f"Session {x_session_id} not found")

    qna_instance, _ = qna_instances[x_session_id]

    try:
        # Extract chat history from QnA instance
        chat_history = qna_instance.chat_history

        # Extract additionals dictionary
        additionals = qna_instance.additionals

        # Prepare payload
        payload = {"chatHistory": chat_history, "additionals": additionals}

        return ServiceResponse(payload=payload, metadata=get_metadata())

    except Exception as e:
        logger.error(f"Error getting session state: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@v1_router.post("/additionals", tags=["context"], response_model=ServiceResponse)
async def add_additional_info(
    additionals_request: AdditionalsRequest,
    x_session_id: str = Header(
        ...,
        alias="X-Session-ID",
        description="Active session ID for which to add additional information",
    ),
):
    """
    Add additional information to a specific session.

    This endpoint adds one or more pieces of additional information to the specified session.
    Each piece of information can be either a direct string or the result of a REST API call.

    Parameters:
        additionals_request: The additional information to add
        x_session_id: Session ID provided in X-Session-ID header

    Returns:
        ServiceResponse containing:
            - payload: Dictionary with information about added items
            - metadata: Request metadata including message ID and timestamp

    Raises:
        HTTPException(404): If the specified session is not found
        HTTPException(409): If the session is currently processing a message
        HTTPException(500): For any processing errors
    """
    if x_session_id not in qna_instances:
        raise HTTPException(status_code=404, detail=f"Session {x_session_id} not found")

    qna_instance, lock = qna_instances[x_session_id]

    if lock.locked():
        raise HTTPException(
            status_code=409, detail="Session is currently processing a message"
        )

    async with lock:
        try:
            new_additions = {}

            for item in additionals_request.items:
                content = item.content

                # If content is a REST call info, make the call
                if not isinstance(content, str):
                    rest_result = make_rest_call(
                        base_url=str(content.base_url),
                        method=content.method,
                        headers=content.headers,
                        params=content.params,
                    )

                    # Add description if provided
                    if item.description:
                        content_text = f"{item.description}\n\n{rest_result}"
                    else:
                        content_text = rest_result
                else:
                    content_text = content

                new_additions[item.id] = content_text

            # Add all new additions to the QnA instance
            qna_instance.append_additional(new_additions)

            payload = {
                "addedItems": list(new_additions.keys()),
                "message": f"Successfully added {len(new_additions)} additional information items",
            }
            return ServiceResponse(payload=payload, metadata=get_metadata())

        except Exception as e:
            logger.error(
                f"Error adding additional information: {str(e)}", exc_info=True
            )
            raise HTTPException(status_code=500, detail=str(e))


@v1_router.delete("/additionals", tags=["context"], response_model=ServiceResponse)
async def remove_additional_info(
    remove_request: RemoveAdditionalsRequest,
    x_session_id: str = Header(
        ...,
        alias="X-Session-ID",
        description="Active session ID for which to remove additional information",
    ),
):
    """
    Remove additional information from a specific session.

    This endpoint removes one or more pieces of additional information from the specified session.

    Parameters:
        remove_request: The IDs of additional information to remove
        x_session_id: Session ID provided in X-Session-ID header

    Returns:
        ServiceResponse containing:
            - payload: Dictionary with information about removed items
            - metadata: Request metadata including message ID and timestamp

    Raises:
        HTTPException(404): If the specified session is not found
        HTTPException(409): If the session is currently processing a message
        HTTPException(500): For any processing errors
    """
    if x_session_id not in qna_instances:
        raise HTTPException(status_code=404, detail=f"Session {x_session_id} not found")

    qna_instance, lock = qna_instances[x_session_id]

    if lock.locked():
        raise HTTPException(
            status_code=409, detail="Session is currently processing a message"
        )

    async with lock:
        try:
            removed_ids = []

            for item_id in remove_request.ids:
                try:
                    qna_instance.remove_additional(item_id)
                    removed_ids.append(item_id)
                except KeyError:
                    # Skip IDs that don't exist
                    logger.warning(f"Additional information ID {item_id} not found")

            payload = {
                "removedItems": removed_ids,
                "message": f"Successfully removed {len(removed_ids)} additional information items",
            }
            return ServiceResponse(payload=payload, metadata=get_metadata())

        except Exception as e:
            logger.error(
                f"Error removing additional information: {str(e)}", exc_info=True
            )
            raise HTTPException(status_code=500, detail=str(e))


@v1_router.delete("/end", tags=["session"])
async def end_conversation(x_session_id: str = Header(..., alias="X-Session-ID")):
    """
    End a conversation session and clean up associated resources.

    Removes the specified session and its associated QnA instance from memory.
    After this operation, the session ID will no longer be valid for other operations.

    Parameters:
        x_session_id: Session ID provided in X-Session-ID header

    Returns:
        204 No Content on successful deletion

    Raises:
        HTTPException(404): If the specified session does not exist
        HTTPException(500): For any errors during session cleanup
    """
    if x_session_id not in qna_instances:
        raise HTTPException(
            status_code=404, detail=f"Session {x_session_id} does not exist"
        )

    try:
        qna_instances.pop(x_session_id)
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Error ending conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Create FastAPI app
app = FastAPI(
    title="FAQ QnA Assistant API",
    description="API for managing FAQ QnA Assistant sessions",
    version=os.getenv("API_VERSION", "1.0.0"),
    openapi_tags=[
        {
            "name": "session",
            "description": "Operations for managing QnA sessions",
        },
        {
            "name": "conversation",
            "description": "Message handling and conversation interactions",
        },
        {
            "name": "context",
            "description": "Managing additional context information for QnA sessions",
        },
        {"name": "health", "description": "API health and status monitoring"},
    ],
    root_path=real_prefix,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include versioned router
app.include_router(v1_router)


@app.get("/", tags=["health"])
async def root_health_check():
    """
    Basic health check endpoint for the root path.

    This endpoint provides a simple way to verify the service is running
    and responding to requests. Unlike the /health endpoint, this endpoint
    returns minimal information and is suitable for basic uptime monitoring.

    Returns:
        dict: A simple status message indicating the service is running
    """
    return {"status": "ok", "message": "Service is running"}


@app.get("/health", tags=["health"], response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to monitor service status and metrics.

    Provides information about the service's health status, active sessions,
    and any potential errors. This endpoint can be used for monitoring and
    health checks by infrastructure systems.

    Returns:
        HealthResponse containing:
            - status: Current health status (HEALTHY/UNHEALTHY)
            - error: Error message if status is UNHEALTHY (null otherwise)
            - version: Current API version
            - timestamp: ISO format timestamp of the health check
            - metrics: Service metrics including:
                - active_sessions: Total number of existing sessions
                - locked_sessions: Number of sessions currently processing messages
    """
    try:
        # Count total sessions and locked sessions
        total_sessions = len(qna_instances)
        locked_sessions = sum(1 for _, lock in qna_instances.values() if lock.locked())

        metrics = HealthMetrics(
            active_sessions=total_sessions, locked_sessions=locked_sessions
        )

        return HealthResponse(
            status=HealthStatus.HEALTHY,
            version=os.getenv("API_VERSION", "1.0.0"),
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return HealthResponse(
            status=HealthStatus.UNHEALTHY,
            error=f"Error Occurred: \n {str(e)}",
            version=os.getenv("API_VERSION", "1.0.0"),
            timestamp=datetime.now().isoformat(),
        )


if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv(override=True)

    host = os.getenv("HTTP_HOST", "0.0.0.0")
    port = int(os.getenv("HTTP_PORT", "8000"))

    logger.info(f"Starting server with prefix: {real_prefix}")
    uvicorn.run(app, host=host, port=port)

"""
Response Generation Router
API endpoints for context-aware response generation
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time

from shared.utils.logger import get_service_logger

router = APIRouter()
logger = get_service_logger("response_gen_router")


# Pydantic models
class ResponseGenerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Input text to generate response for")
    intent: Dict[str, Any] = Field(default_factory=dict, description="Intent context")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    sentiment: Dict[str, Any] = Field(default_factory=dict, description="Sentiment context")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    language: str = Field(default="english", description="Response language")
    style: Optional[str] = Field(None, description="Response style")
    include_alternatives: bool = Field(default=True, description="Include alternative responses")


class ResponseGenerationResponse(BaseModel):
    response: str
    response_type: str
    style: str
    confidence: float
    template_id: Optional[str]
    alternatives: List[str]
    processing_time_ms: float
    metadata: Dict[str, Any]


class BatchResponseRequest(BaseModel):
    requests: List[ResponseGenerationRequest] = Field(..., min_items=1, max_items=50)
    include_alternatives: bool = Field(default=False, description="Include alternatives for batch requests")


# Global variables (will be injected by main app)
response_generator = None


async def get_response_generator():
    """Dependency to get response generator instance"""
    global response_generator
    if response_generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Response generator not available"
        )
    return response_generator


@router.post("/generate", response_model=ResponseGenerationResponse, summary="Generate Response")
async def generate_response(
    request: ResponseGenerationRequest,
    background_tasks: BackgroundTasks,
    generator=Depends(get_response_generator)
):
    """
    Generate a context-aware response for the given input.

    - **text**: Input text to generate response for
    - **intent**: Intent context for better response
    - **entities**: Extracted entities for personalization
    - **sentiment**: Sentiment context for tone adjustment
    - **context**: Additional conversation context
    - **conversation_id**: Conversation ID for context awareness
    - **language**: Response language
    - **style**: Response style (professional, friendly, casual, etc.)
    - **include_alternatives**: Include alternative responses
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input text cannot be empty"
            )

        # Convert style string to enum if provided
        style = None
        if request.style:
            from .response_generator import ResponseStyle
            try:
                style = ResponseStyle(request.style)
            except ValueError:
                logger.warning(
                    "invalid_style_provided",
                    style=request.style,
                    using_default=True
                )

        # Generate response
        result = await generator.generate_response(
            input_text=request.text,
            intent=request.intent,
            entities=request.entities,
            sentiment=request.sentiment,
            context=request.context,
            conversation_id=request.conversation_id,
            language=request.language,
            style=style,
            include_alternatives=request.include_alternatives
        )

        processing_time = time.time() - start_time

        # Log generation result
        background_tasks.add_task(
            logger.info,
            "response_generated",
            text_length=len(request.text),
            response_type=result.get("response_type"),
            confidence=result.get("confidence"),
            language=request.language,
            conversation_id=request.conversation_id,
            processing_time=processing_time
        )

        response = ResponseGenerationResponse(
            response=result.get("response", ""),
            response_type=result.get("response_type", "information"),
            style=result.get("style", "professional"),
            confidence=result.get("confidence", 0.0),
            template_id=result.get("template_id"),
            alternatives=result.get("alternatives", []),
            processing_time_ms=result.get("processing_time_ms", processing_time * 1000),
            metadata=result.get("metadata", {})
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "response_generation_error",
            text_length=len(request.text),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Response generation failed"
        )


@router.post("/generate-batch", summary="Generate Multiple Responses")
async def generate_batch_responses(
    request: BatchResponseRequest,
    background_tasks: BackgroundTasks,
    generator=Depends(get_response_generator)
):
    """
    Generate responses for multiple requests in batch.

    - **requests**: List of response generation requests
    - **include_alternatives**: Include alternative responses
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.requests:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Requests list cannot be empty"
            )

        if len(request.requests) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 requests allowed per batch"
            )

        # Process batch
        results = []
        for req in request.requests:
            try:
                # Convert style string to enum if provided
                style = None
                if req.style:
                    from .response_generator import ResponseStyle
                    try:
                        style = ResponseStyle(req.style)
                    except ValueError:
                        pass

                result = await generator.generate_response(
                    input_text=req.text,
                    intent=req.intent,
                    entities=req.entities,
                    sentiment=req.sentiment,
                    context=req.context,
                    conversation_id=req.conversation_id,
                    language=req.language,
                    style=style,
                    include_alternatives=request.include_alternatives
                )
                results.append({
                    "success": True,
                    "request_id": str(hash(req.text)),
                    "response": result
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "request_id": str(hash(req.text)),
                    "error": str(e)
                })

        processing_time = time.time() - start_time

        # Log batch processing
        background_tasks.add_task(
            logger.info,
            "batch_response_generated",
            requests_count=len(request.requests),
            processing_time=processing_time,
            success_count=len([r for r in results if r.get("success", False)])
        )

        return {
            "results": results,
            "batch_size": len(request.requests),
            "processing_time_ms": processing_time * 1000,
            "success_count": len([r for r in results if r.get("success", False)])
        }

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "batch_response_generation_error",
            requests_count=len(request.requests),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch response generation failed"
        )


@router.get("/templates", summary="Get Response Templates")
async def get_response_templates(
    language: str = "english",
    response_type: Optional[str] = None,
    generator=Depends(get_response_generator)
):
    """
    Get available response templates.
    """
    try:
        templates = generator.template_manager.get_templates_by_criteria(
            language=language,
            response_type=response_type
        )

        return {
            "templates": templates,
            "language": language,
            "response_type": response_type,
            "total_count": len(templates)
        }

    except Exception as e:
        logger.error(
            "get_response_templates_error",
            language=language,
            response_type=response_type,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get response templates"
        )


@router.get("/templates/{template_id}", summary="Get Specific Template")
async def get_response_template(
    template_id: str,
    generator=Depends(get_response_generator)
):
    """
    Get a specific response template by ID.
    """
    try:
        template = generator.template_manager.get_template(template_id)

        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template '{template_id}' not found"
            )

        return {
            "template_id": template.template_id,
            "text": template.text,
            "response_type": template.response_type.value,
            "style": template.style.value,
            "language": template.language,
            "placeholders": template.placeholders,
            "conditions": template.conditions,
            "priority": template.priority
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_response_template_error",
            template_id=template_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get response template"
        )


@router.get("/context/{conversation_id}", summary="Get Conversation Context")
async def get_conversation_context(
    conversation_id: str,
    generator=Depends(get_response_generator)
):
    """
    Get conversation context for a specific conversation ID.
    """
    try:
        context = generator.context_manager.get_context(conversation_id)

        return {
            "conversation_id": conversation_id,
            "context": context,
            "has_history": len(context.get("history", [])) > 0,
            "message_count": context.get("response_count", 0),
            "recent_entities": generator.context_manager.get_recent_entities(conversation_id)
        }

    except Exception as e:
        logger.error(
            "get_conversation_context_error",
            conversation_id=conversation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get conversation context"
        )


@router.delete("/context/{conversation_id}", summary="Clear Conversation Context")
async def clear_conversation_context(
    conversation_id: str,
    generator=Depends(get_response_generator)
):
    """
    Clear conversation context for a specific conversation ID.
    """
    try:
        # Remove context if it exists
        if conversation_id in generator.context_manager.conversation_contexts:
            del generator.context_manager.conversation_contexts[conversation_id]
            return {"message": "Conversation context cleared", "conversation_id": conversation_id}
        else:
            return {"message": "No context found to clear", "conversation_id": conversation_id}

    except Exception as e:
        logger.error(
            "clear_conversation_context_error",
            conversation_id=conversation_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear conversation context"
        )


@router.post("/templates", summary="Add Response Template")
async def add_response_template(
    template_data: Dict[str, Any],
    generator=Depends(get_response_generator)
):
    """
    Add a new response template.
    """
    try:
        from .response_generator import ResponseTemplate, ResponseType, ResponseStyle

        # Validate required fields
        required_fields = ["template_id", "text", "response_type", "style", "language"]
        for field in required_fields:
            if field not in template_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )

        # Create template
        template = ResponseTemplate(
            template_id=template_data["template_id"],
            text=template_data["text"],
            response_type=ResponseType(template_data["response_type"]),
            style=ResponseStyle(template_data["style"]),
            language=template_data["language"],
            placeholders=template_data.get("placeholders", []),
            conditions=template_data.get("conditions", {}),
            priority=template_data.get("priority", 1)
        )

        # Add template
        generator.template_manager.templates[template.template_id] = template

        return {
            "message": "Template added successfully",
            "template_id": template.template_id,
            "template": {
                "template_id": template.template_id,
                "response_type": template.response_type.value,
                "style": template.style.value,
                "language": template.language
            }
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid value: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "add_response_template_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add response template"
        )


@router.get("/health", summary="Response Generator Health")
async def response_generator_health(generator=Depends(get_response_generator)):
    """
    Check the health status of the response generator.
    """
    try:
        return {
            "service": "response_generator",
            "healthy": generator.is_ready(),
            "templates_count": len(generator.template_manager.templates),
            "active_conversations": len(generator.context_manager.conversation_contexts),
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(
            "response_generator_health_error",
            error=str(e),
            exc_info=True
        )
        return {
            "service": "response_generator",
            "healthy": False,
            "error": str(e),
            "timestamp": time.time()
        }
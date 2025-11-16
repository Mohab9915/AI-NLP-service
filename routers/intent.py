"""
Intent Recognition Router
API endpoints for intent classification and recognition
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time

from shared.utils.logger import get_service_logger

router = APIRouter()
logger = get_service_logger("intent_router")


# Pydantic models
class IntentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    language: str = Field(default="english", description="Language code")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    use_patterns: bool = Field(default=True, description="Use pattern-based classification")
    use_ai: bool = Field(default=True, description="Use AI-based classification")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")


class BatchIntentRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    language: str = Field(default="english", description="Language code")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    use_patterns: bool = Field(default=True, description="Use pattern-based classification")
    use_ai: bool = Field(default=True, description="Use AI-based classification")


class IntentResponse(BaseModel):
    intent: str
    confidence: float
    alternative_intents: List[Dict[str, Any]]
    language_detected: str
    processing_time_ms: float
    pattern_matched: Optional[str] = None
    ai_used: bool


# Global variables (will be injected by main app)
intent_classifier = None


async def get_intent_classifier():
    """Dependency to get intent classifier instance"""
    global intent_classifier
    if intent_classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Intent classifier not available"
        )
    return intent_classifier


@router.post("/classify", response_model=IntentResponse, summary="Classify Text Intent")
async def classify_intent(
    request: IntentRequest,
    background_tasks: BackgroundTasks,
    classifier=Depends(get_intent_classifier)
):
    """
    Classify the intent of a single text.

    - **text**: The text to classify for intent
    - **language**: Target language (english, arabic, hebrew)
    - **context**: Additional context information
    - **use_patterns**: Enable pattern-based classification
    - **use_ai**: Enable AI-based classification
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        # Classify intent
        result = await classifier.classify_intent(
            text=request.text,
            context=request.context,
            language=request.language,
            use_patterns=request.use_patterns,
            use_ai=request.use_ai
        )

        processing_time = time.time() - start_time

        # Log classification result
        background_tasks.add_task(
            logger.info,
            "intent_classified",
            text_length=len(request.text),
            intent=result.get("intent"),
            confidence=result.get("confidence"),
            language=request.language,
            processing_time=processing_time
        )

        response = IntentResponse(
            intent=result.get("intent", "unknown"),
            confidence=result.get("confidence", 0.0),
            alternative_intents=result.get("alternative_intents", []),
            language_detected=result.get("language_detected", request.language),
            processing_time_ms=processing_time * 1000,
            pattern_matched=result.get("pattern_matched"),
            ai_used=result.get("ai_used", False)
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "intent_classification_error",
            text_length=len(request.text),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Intent classification failed"
        )


@router.post("/classify-batch", summary="Classify Multiple Text Intents")
async def classify_batch_intents(
    request: BatchIntentRequest,
    background_tasks: BackgroundTasks,
    classifier=Depends(get_intent_classifier)
):
    """
    Classify intents for multiple texts in batch.

    - **texts**: List of texts to classify
    - **language**: Target language
    - **context**: Additional context information
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Texts list cannot be empty"
            )

        if len(request.texts) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 texts allowed per batch"
            )

        # Process batch
        results = await classifier.batch_classify_intents(
            texts=request.texts,
            context=request.context,
            language=request.language,
            use_patterns=request.use_patterns,
            use_ai=request.use_ai
        )

        processing_time = time.time() - start_time

        # Log batch processing
        background_tasks.add_task(
            logger.info,
            "batch_intent_classified",
            texts_count=len(request.texts),
            processing_time=processing_time,
            success_count=len([r for r in results if r.get("success", False)])
        )

        return {
            "results": results,
            "batch_size": len(request.texts),
            "language": request.language,
            "processing_time_ms": processing_time * 1000,
            "success_count": len([r for r in results if r.get("success", False)])
        }

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "batch_intent_classification_error",
            texts_count=len(request.texts),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch intent classification failed"
        )


@router.get("/intents", summary="Get Available Intents")
async def get_available_intents(classifier=Depends(get_intent_classifier)):
    """
    Get list of available intents that the classifier can recognize.
    """
    try:
        intents = classifier.get_available_intents()

        return {
            "available_intents": intents,
            "total_count": len(intents),
            "supported_languages": classifier.get_supported_languages()
        }

    except Exception as e:
        logger.error(
            "get_available_intents_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available intents"
        )


@router.get("/intents/{intent_name}", summary="Get Intent Details")
async def get_intent_details(
    intent_name: str,
    classifier=Depends(get_intent_classifier)
):
    """
    Get detailed information about a specific intent.
    """
    try:
        intent_info = classifier.get_intent_details(intent_name)

        if not intent_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Intent '{intent_name}' not found"
            )

        return intent_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_intent_details_error",
            intent_name=intent_name,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get intent details"
        )


@router.get("/patterns", summary="Get Intent Patterns")
async def get_intent_patterns(
    language: str = "english",
    classifier=Depends(get_intent_classifier)
):
    """
    Get intent patterns used for pattern-based classification.
    """
    try:
        patterns = classifier.get_intent_patterns(language)

        return {
            "language": language,
            "patterns": patterns,
            "total_patterns": len(patterns)
        }

    except Exception as e:
        logger.error(
            "get_intent_patterns_error",
            language=language,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get intent patterns"
        )


@router.post("/train", summary="Train Intent Classifier")
async def train_intent_classifier(
    training_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    classifier=Depends(get_intent_classifier)
):
    """
    Train the intent classifier with new data.
    """
    start_time = time.time()

    try:
        # Validate training data
        if "examples" not in training_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training data must contain 'examples' field"
            )

        examples = training_data["examples"]
        if not isinstance(examples, list) or len(examples) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Examples must be a non-empty list"
            )

        # Start training in background
        background_tasks.add_task(
            _train_classifier_background,
            classifier,
            training_data
        )

        processing_time = time.time() - start_time

        return {
            "message": "Training started in background",
            "examples_count": len(examples),
            "processing_time_ms": processing_time * 1000,
            "estimated_duration": "2-5 minutes"
        }

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "train_intent_classifier_error",
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start intent classifier training"
        )


@router.get("/health", summary="Intent Classifier Health")
async def intent_classifier_health(classifier=Depends(get_intent_classifier)):
    """
    Check the health status of the intent classifier.
    """
    try:
        health_status = await classifier.health_check()

        return {
            "service": "intent_classifier",
            "healthy": health_status.get("healthy", False),
            "details": health_status,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(
            "intent_classifier_health_error",
            error=str(e),
            exc_info=True
        )
        return {
            "service": "intent_classifier",
            "healthy": False,
            "error": str(e),
            "timestamp": time.time()
        }


async def _train_classifier_background(classifier, training_data: Dict[str, Any]):
    """Background task for training the classifier"""
    try:
        logger.info(
            "intent_classifier_training_started",
            examples_count=len(training_data.get("examples", []))
        )

        # Train the classifier
        success = await classifier.train(training_data)

        if success:
            logger.info("intent_classifier_training_completed_successfully")
        else:
            logger.error("intent_classifier_training_failed")

    except Exception as e:
        logger.error(
            "intent_classifier_training_background_error",
            error=str(e),
            exc_info=True
        )
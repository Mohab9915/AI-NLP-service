"""
Entity Extraction Router
API endpoints for entity extraction and recognition
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time

from shared.utils.logger import get_service_logger

router = APIRouter()
logger = get_service_logger("entities_router")


# Pydantic models
class EntityRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to extract entities from")
    language: str = Field(default="english", description="Language code")
    intent: Dict[str, Any] = Field(default_factory=dict, description="Intent context")
    entity_types: Optional[List[str]] = Field(None, description="Specific entity types to extract")
    use_patterns: bool = Field(default=True, description="Use pattern-based extraction")
    use_nlp: bool = Field(default=True, description="Use NLP model extraction")


class BatchEntityRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to process")
    language: str = Field(default="english", description="Language code")
    intent: Dict[str, Any] = Field(default_factory=dict, description="Intent context")
    entity_types: Optional[List[str]] = Field(None, description="Specific entity types to extract")
    use_patterns: bool = Field(default=True, description="Use pattern-based extraction")
    use_nlp: bool = Field(default=True, description="Use NLP model extraction")


class EntityResponse(BaseModel):
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    language_detected: str
    processing_time_ms: float
    pattern_entities_count: int = 0
    nlp_entities_count: int = 0


# Global variables (will be injected by main app)
entity_extractor = None


async def get_entity_extractor():
    """Dependency to get entity extractor instance"""
    global entity_extractor
    if entity_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Entity extractor not available"
        )
    return entity_extractor


@router.post("/extract", response_model=EntityResponse, summary="Extract Entities from Text")
async def extract_entities(
    request: EntityRequest,
    background_tasks: BackgroundTasks,
    extractor=Depends(get_entity_extractor)
):
    """
    Extract entities from a single text.

    - **text**: The text to extract entities from
    - **language**: Target language (english, arabic, hebrew)
    - **intent**: Intent context for better extraction
    - **entity_types**: Specific entity types to extract
    - **use_patterns**: Enable pattern-based extraction
    - **use_nlp**: Enable NLP model extraction
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        # Extract entities
        result = await extractor.extract_entities(
            text=request.text,
            intent=request.intent,
            language=request.language,
            entity_types=request.entity_types,
            use_patterns=request.use_patterns,
            use_nlp=request.use_nlp
        )

        processing_time = time.time() - start_time

        # Count entity types
        pattern_entities_count = len([
            e for e in result.get("entities", [])
            if e.get("extraction_method") == "pattern"
        ])
        nlp_entities_count = len([
            e for e in result.get("entities", [])
            if e.get("extraction_method") == "nlp"
        ])

        # Log extraction result
        background_tasks.add_task(
            logger.info,
            "entities_extracted",
            text_length=len(request.text),
            entities_count=len(result.get("entities", [])),
            relations_count=len(result.get("relations", [])),
            language=request.language,
            processing_time=processing_time
        )

        response = EntityResponse(
            entities=result.get("entities", []),
            relations=result.get("relations", []),
            language_detected=result.get("language_detected", request.language),
            processing_time_ms=processing_time * 1000,
            pattern_entities_count=pattern_entities_count,
            nlp_entities_count=nlp_entities_count
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "entity_extraction_error",
            text_length=len(request.text),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Entity extraction failed"
        )


@router.post("/extract-batch", summary="Extract Entities from Multiple Texts")
async def extract_batch_entities(
    request: BatchEntityRequest,
    background_tasks: BackgroundTasks,
    extractor=Depends(get_entity_extractor)
):
    """
    Extract entities from multiple texts in batch.

    - **texts**: List of texts to extract entities from
    - **language**: Target language
    - **intent**: Intent context for better extraction
    - **entity_types**: Specific entity types to extract
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
        results = await extractor.batch_extract_entities(
            texts=request.texts,
            intent=request.intent,
            language=request.language,
            entity_types=request.entity_types,
            use_patterns=request.use_patterns,
            use_nlp=request.use_nlp
        )

        processing_time = time.time() - start_time

        # Log batch processing
        background_tasks.add_task(
            logger.info,
            "batch_entities_extracted",
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
            "batch_entity_extraction_error",
            texts_count=len(request.texts),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch entity extraction failed"
        )


@router.get("/types", summary="Get Supported Entity Types")
async def get_supported_entity_types(extractor=Depends(get_entity_extractor)):
    """
    Get list of supported entity types that the extractor can recognize.
    """
    try:
        entity_types = extractor.get_supported_entity_types()

        return {
            "supported_entity_types": entity_types,
            "total_count": len(entity_types),
            "supported_languages": extractor.get_supported_languages()
        }

    except Exception as e:
        logger.error(
            "get_supported_entity_types_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get supported entity types"
        )


@router.get("/types/{entity_type}", summary="Get Entity Type Details")
async def get_entity_type_details(
    entity_type: str,
    language: str = "english",
    extractor=Depends(get_entity_extractor)
):
    """
    Get detailed information about a specific entity type.
    """
    try:
        entity_info = extractor.get_entity_type_details(entity_type, language)

        if not entity_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entity type '{entity_type}' not found"
            )

        return entity_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_entity_type_details_error",
            entity_type=entity_type,
            language=language,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get entity type details"
        )


@router.get("/patterns", summary="Get Entity Patterns")
async def get_entity_patterns(
    language: str = "english",
    extractor=Depends(get_entity_extractor)
):
    """
    Get entity patterns used for pattern-based extraction.
    """
    try:
        patterns = extractor.get_entity_patterns(language)

        return {
            "language": language,
            "patterns": patterns,
            "total_patterns": len(patterns)
        }

    except Exception as e:
        logger.error(
            "get_entity_patterns_error",
            language=language,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get entity patterns"
        )


@router.post("/relations", summary="Extract Entity Relations")
async def extract_entity_relations(
    text: str,
    entities: List[Dict[str, Any]],
    language: str = "english",
    extractor=Depends(get_entity_extractor)
):
    """
    Extract relations between entities in text.
    """
    try:
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        if not entities:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Entities list cannot be empty"
            )

        relations = await extractor.extract_relations(
            text=text,
            entities=entities,
            language=language
        )

        return {
            "relations": relations,
            "entities_count": len(entities),
            "relations_count": len(relations),
            "language": language
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "entity_relations_extraction_error",
            text_length=len(text),
            entities_count=len(entities),
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Entity relation extraction failed"
        )


@router.post("/train", summary="Train Entity Extractor")
async def train_entity_extractor(
    training_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    extractor=Depends(get_entity_extractor)
):
    """
    Train the entity extractor with new data.
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
            _train_extractor_background,
            extractor,
            training_data
        )

        processing_time = time.time() - start_time

        return {
            "message": "Training started in background",
            "examples_count": len(examples),
            "processing_time_ms": processing_time * 1000,
            "estimated_duration": "3-7 minutes"
        }

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "train_entity_extractor_error",
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start entity extractor training"
        )


@router.get("/health", summary="Entity Extractor Health")
async def entity_extractor_health(extractor=Depends(get_entity_extractor)):
    """
    Check the health status of the entity extractor.
    """
    try:
        health_status = await extractor.health_check()

        return {
            "service": "entity_extractor",
            "healthy": health_status.get("healthy", False),
            "details": health_status,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(
            "entity_extractor_health_error",
            error=str(e),
            exc_info=True
        )
        return {
            "service": "entity_extractor",
            "healthy": False,
            "error": str(e),
            "timestamp": time.time()
        }


async def _train_extractor_background(extractor, training_data: Dict[str, Any]):
    """Background task for training the extractor"""
    try:
        logger.info(
            "entity_extractor_training_started",
            examples_count=len(training_data.get("examples", []))
        )

        # Train the extractor
        success = await extractor.train(training_data)

        if success:
            logger.info("entity_extractor_training_completed_successfully")
        else:
            logger.error("entity_extractor_training_failed")

    except Exception as e:
        logger.error(
            "entity_extractor_training_background_error",
            error=str(e),
            exc_info=True
        )
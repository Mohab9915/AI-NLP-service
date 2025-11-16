"""
Language Detection Router
API endpoints for language detection and text processing
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time

from shared.utils.logger import get_service_logger

router = APIRouter()
logger = get_service_logger("language_router")


# Pydantic models
class LanguageDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to detect language")
    fallback_language: str = Field(default="english", description="Fallback language if detection fails")
    confidence_threshold: float = Field(default=0.5, description="Minimum confidence threshold")


class LanguageDetectionResponse(BaseModel):
    detected_language: str
    confidence: float
    supported: bool
    alternatives: List[Dict[str, Any]]
    processing_time_ms: float
    text_sample: str


class TextProcessingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to process")
    language: Optional[str] = Field(None, description="Language code (auto-detected if not provided)")
    operations: List[str] = Field(default=["normalize", "tokenize", "clean"], description="Processing operations")


class TextProcessingResponse(BaseModel):
    original_text: str
    processed_text: str
    detected_language: str
    operations_performed: List[str]
    tokens: List[str]
    sentences: List[str]
    processing_time_ms: float


# Global variables (will be injected by main app)
text_processor = None


async def get_text_processor():
    """Dependency to get text processor instance"""
    global text_processor
    if text_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Text processor not available"
        )
    return text_processor


@router.post("/detect", response_model=LanguageDetectionResponse, summary="Detect Text Language")
async def detect_language(
    request: LanguageDetectionRequest,
    background_tasks: BackgroundTasks,
    processor=Depends(get_text_processor)
):
    """
    Detect the language of a single text.

    - **text**: The text to detect language for
    - **fallback_language**: Language to use if detection fails
    - **confidence_threshold**: Minimum confidence threshold
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        # Detect language
        result = await processor.detect_language(
            text=request.text,
            fallback_language=request.fallback_language,
            confidence_threshold=request.confidence_threshold
        )

        processing_time = time.time() - start_time

        # Log detection result
        background_tasks.add_task(
            logger.info,
            "language_detected",
            text_length=len(request.text),
            detected_language=result.get("detected_language"),
            confidence=result.get("confidence"),
            processing_time=processing_time
        )

        response = LanguageDetectionResponse(
            detected_language=result.get("detected_language", request.fallback_language),
            confidence=result.get("confidence", 0.0),
            supported=result.get("supported", True),
            alternatives=result.get("alternatives", []),
            processing_time_ms=processing_time * 1000,
            text_sample=request.text[:100] + "..." if len(request.text) > 100 else request.text
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "language_detection_error",
            text_length=len(request.text),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Language detection failed"
        )


@router.post("/process", response_model=TextProcessingResponse, summary="Process Text")
async def process_text(
    request: TextProcessingRequest,
    background_tasks: BackgroundTasks,
    processor=Depends(get_text_processor)
):
    """
    Process text with various operations.

    - **text**: The text to process
    - **language**: Language code (auto-detected if not provided)
    - **operations**: List of processing operations to perform
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        if not request.operations:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Operations list cannot be empty"
            )

        # Process text
        result = await processor.process_text(
            text=request.text,
            language=request.language,
            operations=request.operations
        )

        processing_time = time.time() - start_time

        # Log processing result
        background_tasks.add_task(
            logger.info,
            "text_processed",
            text_length=len(request.text),
            operations_count=len(request.operations),
            language=result.get("detected_language"),
            processing_time=processing_time
        )

        response = TextProcessingResponse(
            original_text=request.text,
            processed_text=result.get("processed_text", ""),
            detected_language=result.get("detected_language", "unknown"),
            operations_performed=result.get("operations_performed", []),
            tokens=result.get("tokens", []),
            sentences=result.get("sentences", []),
            processing_time_ms=processing_time * 1000
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "text_processing_error",
            text_length=len(request.text),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text processing failed"
        )


@router.post("/detect-batch", summary="Detect Languages for Multiple Texts")
async def detect_batch_languages(
    texts: List[str],
    fallback_language: str = "english",
    confidence_threshold: float = 0.5,
    background_tasks: BackgroundTasks,
    processor=Depends(get_text_processor)
):
    """
    Detect languages for multiple texts in batch.

    - **texts**: List of texts to detect language for
    - **fallback_language**: Language to use if detection fails
    - **confidence_threshold**: Minimum confidence threshold
    """
    start_time = time.time()

    try:
        # Validate input
        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Texts list cannot be empty"
            )

        if len(texts) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 texts allowed per batch"
            )

        # Process batch
        results = await processor.batch_detect_languages(
            texts=texts,
            fallback_language=fallback_language,
            confidence_threshold=confidence_threshold
        )

        processing_time = time.time() - start_time

        # Log batch processing
        background_tasks.add_task(
            logger.info,
            "batch_language_detected",
            texts_count=len(texts),
            processing_time=processing_time,
            success_count=len([r for r in results if r.get("success", False)])
        )

        return {
            "results": results,
            "batch_size": len(texts),
            "fallback_language": fallback_language,
            "confidence_threshold": confidence_threshold,
            "processing_time_ms": processing_time * 1000,
            "success_count": len([r for r in results if r.get("success", False)])
        }

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "batch_language_detection_error",
            texts_count=len(texts),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch language detection failed"
        )


@router.post("/tokenize", summary="Tokenize Text")
async def tokenize_text(
    text: str,
    language: str = "english",
    processor=Depends(get_text_processor)
):
    """
    Tokenize text into words/tokens.
    """
    try:
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        tokens = await processor.tokenize_text(text, language)

        return {
            "tokens": tokens,
            "token_count": len(tokens),
            "original_text": text,
            "language": language
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "text_tokenization_error",
            text_length=len(text),
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text tokenization failed"
        )


@router.post("/sentences", summary="Split Text into Sentences")
async def split_sentences(
    text: str,
    language: str = "english",
    processor=Depends(get_text_processor)
):
    """
    Split text into sentences.
    """
    try:
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        sentences = await processor.split_sentences(text, language)

        return {
            "sentences": sentences,
            "sentence_count": len(sentences),
            "original_text": text,
            "language": language
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "sentence_splitting_error",
            text_length=len(text),
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sentence splitting failed"
        )


@router.post("/normalize", summary="Normalize Text")
async def normalize_text(
    text: str,
    language: str = "english",
    processor=Depends(get_text_processor)
):
    """
    Normalize text (cleaning, standardization, etc.).
    """
    try:
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        normalized_text = await processor.normalize_text(text, language)

        return {
            "original_text": text,
            "normalized_text": normalized_text,
            "language": language,
            "changes_made": len(text) != len(normalized_text)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "text_normalization_error",
            text_length=len(text),
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text normalization failed"
        )


@router.get("/supported", summary="Get Supported Languages")
async def get_supported_languages(processor=Depends(get_text_processor)):
    """
    Get list of supported languages.
    """
    try:
        languages = processor.get_supported_languages()

        return {
            "supported_languages": languages,
            "total_count": len(languages),
            "language_codes": [lang.get("code") for lang in languages]
        }

    except Exception as e:
        logger.error(
            "get_supported_languages_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get supported languages"
        )


@router.get("/operations", summary="Get Supported Operations")
async def get_supported_operations(processor=Depends(get_text_processor)):
    """
    Get list of supported text processing operations.
    """
    try:
        operations = processor.get_supported_operations()

        return {
            "supported_operations": operations,
            "total_count": len(operations)
        }

    except Exception as e:
        logger.error(
            "get_supported_operations_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get supported operations"
        )


@router.get("/health", summary="Text Processor Health")
async def text_processor_health(processor=Depends(get_text_processor)):
    """
    Check the health status of the text processor.
    """
    try:
        health_status = await processor.health_check()

        return {
            "service": "text_processor",
            "healthy": health_status.get("healthy", False),
            "details": health_status,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(
            "text_processor_health_error",
            error=str(e),
            exc_info=True
        )
        return {
            "service": "text_processor",
            "healthy": False,
            "error": str(e),
            "timestamp": time.time()
        }
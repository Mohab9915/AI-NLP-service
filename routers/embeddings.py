"""
Embeddings Router
API endpoints for text embeddings generation and similarity search
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time

from shared.utils.logger import get_service_logger

router = APIRouter()
logger = get_service_logger("embeddings_router")


# Pydantic models
class EmbeddingGenerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to generate embeddings for")
    language: str = Field(default="english", description="Language code")
    model: Optional[str] = Field(None, description="Embedding model to use")


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to process")
    language: str = Field(default="english", description="Language code")
    model: Optional[str] = Field(None, description="Embedding model to use")
    batch_size: int = Field(default=32, description="Batch size for processing")


class SimilaritySearchRequest(BaseModel):
    query_text: str = Field(..., min_length=1, max_length=1000, description="Query text for similarity search")
    language: str = Field(default="english", description="Language code")
    top_k: int = Field(default=10, description="Number of similar texts to return")
    threshold: float = Field(default=0.5, description="Minimum similarity threshold")
    distance_metric: str = Field(default="cosine", description="Distance metric: cosine, euclidean, manhattan, dot_product")
    include_embeddings: bool = Field(default=False, description="Include embedding vectors in results")


class SimilarityCalculationRequest(BaseModel):
    text1: str = Field(..., min_length=1, max_length=1000, description="First text")
    text2: str = Field(..., min_length=1, max_length=1000, description="Second text")
    language: str = Field(default="english", description="Language code")
    distance_metric: str = Field(default="cosine", description="Distance metric")


# Global variables (will be injected by main app)
embeddings_manager = None


async def get_embeddings_manager():
    """Dependency to get embeddings manager instance"""
    global embeddings_manager
    if embeddings_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embeddings manager not available"
        )
    return embeddings_manager


@router.post("/generate", summary="Generate Text Embeddings")
async def generate_embeddings(
    request: EmbeddingGenerationRequest,
    background_tasks: BackgroundTasks,
    manager=Depends(get_embeddings_manager)
):
    """
    Generate embeddings for a single text.

    - **text**: Text to generate embeddings for
    - **language**: Language code
    - **model**: Embedding model to use
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        # Convert model string to enum if provided
        model = None
        if request.model:
            from .embeddings_manager import EmbeddingModel
            try:
                model = EmbeddingModel(request.model)
            except ValueError:
                logger.warning(
                    "invalid_model_provided",
                    model=request.model,
                    using_default=True
                )

        # Generate embeddings
        result = await manager.generate_embeddings(
            text=request.text,
            language=request.language,
            model=model
        )

        processing_time = time.time() - start_time

        # Log generation result
        background_tasks.add_task(
            logger.info,
            "embeddings_generated",
            text_length=len(request.text),
            dimension=result.get("dimension", 0),
            model=result.get("model", "unknown"),
            cached=result.get("cached", False),
            processing_time=processing_time
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "embeddings_generation_error",
            text_length=len(request.text),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embeddings generation failed"
        )


@router.post("/generate-batch", summary="Generate Batch Embeddings")
async def generate_batch_embeddings(
    request: BatchEmbeddingRequest,
    background_tasks: BackgroundTasks,
    manager=Depends(get_embeddings_manager)
):
    """
    Generate embeddings for multiple texts in batch.

    - **texts**: List of texts to process
    - **language**: Language code
    - **model**: Embedding model to use
    - **batch_size**: Batch size for processing
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

        # Convert model string to enum if provided
        model = None
        if request.model:
            from .embeddings_manager import EmbeddingModel
            try:
                model = EmbeddingModel(request.model)
            except ValueError:
                pass

        # Generate batch embeddings
        results = await manager.generate_batch_embeddings(
            texts=request.texts,
            language=request.language,
            model=model,
            batch_size=request.batch_size
        )

        processing_time = time.time() - start_time

        # Log batch processing
        background_tasks.add_task(
            logger.info,
            "batch_embeddings_generated",
            texts_count=len(request.texts),
            processing_time=processing_time,
            success_count=len([r for r in results if r.get("embeddings")])
        )

        return {
            "results": results,
            "batch_size": len(request.texts),
            "language": request.language,
            "processing_time_ms": processing_time * 1000,
            "success_count": len([r for r in results if r.get("embeddings")])
        }

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "batch_embeddings_generation_error",
            texts_count=len(request.texts),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch embeddings generation failed"
        )


@router.post("/similarity/search", summary="Search Similar Texts")
async def search_similar_texts(
    request: SimilaritySearchRequest,
    background_tasks: BackgroundTasks,
    manager=Depends(get_embeddings_manager)
):
    """
    Search for texts similar to the query text.

    - **query_text**: Query text for similarity search
    - **language**: Language code
    - **top_k**: Number of similar texts to return
    - **threshold**: Minimum similarity threshold
    - **distance_metric**: Distance metric to use
    - **include_embeddings**: Include embedding vectors in results
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.query_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query text cannot be empty"
            )

        # Convert distance metric string to enum
        from .embeddings_manager import VectorDistance
        try:
            distance_metric = VectorDistance(request.distance_metric)
        except ValueError:
            distance_metric = VectorDistance.COSINE
            logger.warning(
                "invalid_distance_metric",
                metric=request.distance_metric,
                using_default="cosine"
            )

        # Search for similar texts
        result = await manager.search_similar_texts(
            query_text=request.query_text,
            language=request.language,
            top_k=request.top_k,
            threshold=request.threshold,
            distance_metric=distance_metric,
            include_embeddings=request.include_embeddings
        )

        processing_time = time.time() - start_time

        # Log search result
        background_tasks.add_task(
            logger.info,
            "similarity_search_completed",
            query_text_length=len(request.query_text),
            similar_texts_found=len(result.get("similar_texts", [])),
            threshold=request.threshold,
            processing_time=processing_time
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "similarity_search_error",
            query_text_length=len(request.query_text),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Similarity search failed"
        )


@router.post("/similarity/calculate", summary="Calculate Text Similarity")
async def calculate_similarity(
    request: SimilarityCalculationRequest,
    background_tasks: BackgroundTasks,
    manager=Depends(get_embeddings_manager)
):
    """
    Calculate similarity between two texts.

    - **text1**: First text
    - **text2**: Second text
    - **language**: Language code
    - **distance_metric**: Distance metric to use
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.text1.strip() or not request.text2.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both texts cannot be empty"
            )

        # Convert distance metric string to enum
        from .embeddings_manager import VectorDistance
        try:
            distance_metric = VectorDistance(request.distance_metric)
        except ValueError:
            distance_metric = VectorDistance.COSINE
            logger.warning(
                "invalid_distance_metric",
                metric=request.distance_metric,
                using_default="cosine"
            )

        # Calculate similarity
        result = await manager.calculate_similarity(
            text1=request.text1,
            text2=request.text2,
            language=request.language,
            distance_metric=distance_metric
        )

        processing_time = time.time() - start_time

        # Log calculation result
        background_tasks.add_task(
            logger.info,
            "similarity_calculated",
            text1_length=len(request.text1),
            text2_length=len(request.text2),
            similarity_score=result.get("similarity_score", 0.0),
            processing_time=processing_time
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "similarity_calculation_error",
            text1_length=len(request.text1),
            text2_length=len(request.text2),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Similarity calculation failed"
        )


@router.delete("/embeddings/{embedding_id}", summary="Delete Embedding")
async def delete_embedding(
    embedding_id: str,
    background_tasks: BackgroundTasks,
    manager=Depends(get_embeddings_manager)
):
    """
    Delete an embedding from the database.
    """
    try:
        result = await manager.delete_embedding(embedding_id)

        # Log deletion result
        background_tasks.add_task(
            logger.info,
            "embedding_deletion_attempted",
            embedding_id=embedding_id,
            deleted=result.get("deleted", False)
        )

        return result

    except Exception as e:
        logger.error(
            "embedding_deletion_error",
            embedding_id=embedding_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding deletion failed"
        )


@router.get("/stats", summary="Get Embeddings Statistics")
async def get_embeddings_stats(manager=Depends(get_embeddings_manager)):
    """
    Get comprehensive statistics about the embeddings system.
    """
    try:
        stats = await manager.get_embeddings_stats()
        return stats

    except Exception as e:
        logger.error(
            "get_embeddings_stats_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get embeddings statistics"
        )


@router.get("/models", summary="Get Available Embedding Models")
async def get_available_models(manager=Depends(get_embeddings_manager)):
    """
    Get list of available embedding models.
    """
    try:
        from .embeddings_manager import EmbeddingModel
        models = list(EmbeddingModel)

        model_details = []
        for model in models:
            config = manager.models.get(model, {})
            model_details.append({
                "name": model.value,
                "dimension": config.get("dimension", 0),
                "languages": config.get("languages", []),
                "max_length": config.get("max_length", 0)
            })

        return {
            "available_models": model_details,
            "total_count": len(models),
            "active_model": manager.active_model.value
        }

    except Exception as e:
        logger.error(
            "get_available_models_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available models"
        )


@router.post("/sync", summary="Synchronize Embeddings")
async def sync_embeddings(
    background_tasks: BackgroundTasks,
    manager=Depends(get_embeddings_manager)
):
    """
    Synchronize embeddings with external storage.
    """
    try:
        # Start sync in background
        background_tasks.add_task(manager.sync_embeddings)

        return {
            "message": "Embeddings synchronization started in background",
            "estimated_duration": "1-5 minutes"
        }

    except Exception as e:
        logger.error(
            "sync_embeddings_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start embeddings synchronization"
        )


@router.get("/health", summary="Embeddings Manager Health")
async def embeddings_manager_health(manager=Depends(get_embeddings_manager)):
    """
    Check the health status of the embeddings manager.
    """
    try:
        return {
            "service": "embeddings_manager",
            "healthy": manager.is_ready(),
            "connected": manager.is_connected(),
            "active_model": manager.active_model.value,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(
            "embeddings_manager_health_error",
            error=str(e),
            exc_info=True
        )
        return {
            "service": "embeddings_manager",
            "healthy": False,
            "connected": False,
            "error": str(e),
            "timestamp": time.time()
        }
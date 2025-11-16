"""
AI & NLP Service - Advanced Natural Language Processing Service
Handles intent recognition, entity extraction, sentiment analysis, and response generation
"""
from fastapi import FastAPI, Request, HTTPException, status, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Union
import httpx

from shared.config.settings import get_settings
from shared.utils.logger import get_service_logger, log_requests
from shared.models.database import ConversationContext
from .routers import intent, entities, sentiment, language, response_gen, health, embeddings
from .nlp_engine import NLPEngine
from .text_processor import TextProcessor
from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .sentiment_analyzer import SentimentAnalyzer
from .response_generator import ResponseGenerator
from .embeddings_manager import EmbeddingsManager
from .cache_manager import CacheManager
from .exceptions import NLPException, setup_exception_handlers

# Initialize settings and logger
settings = get_settings("ai-nlp-service")
logger = get_service_logger("ai-nlp-service")

# Global services
nlp_engine: Optional[NLPEngine] = None
text_processor: Optional[TextProcessor] = None
intent_classifier: Optional[IntentClassifier] = None
entity_extractor: Optional[EntityExtractor] = None
sentiment_analyzer: Optional[SentimentAnalyzer] = None
response_generator: Optional[ResponseGenerator] = None
embeddings_manager: Optional[EmbeddingsManager] = None
cache_manager: Optional[CacheManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global nlp_engine, text_processor, intent_classifier, entity_extractor
    global sentiment_analyzer, response_generator, embeddings_manager, cache_manager

    logger.info("ai_nlp_service_startup", version="1.0.0")

    try:
        # Initialize cache manager first
        cache_manager = CacheManager(settings)
        await cache_manager.initialize()
        logger.info("cache_manager_initialized")

        # Initialize text processor
        text_processor = TextProcessor(settings)
        await text_processor.initialize()
        logger.info("text_processor_initialized")

        # Initialize embeddings manager
        embeddings_manager = EmbeddingsManager(settings)
        await embeddings_manager.initialize()
        logger.info("embeddings_manager_initialized")

        # Initialize NLP components
        intent_classifier = IntentClassifier(settings)
        await intent_classifier.initialize()
        logger.info("intent_classifier_initialized")

        entity_extractor = EntityExtractor(settings)
        await entity_extractor.initialize()
        logger.info("entity_extractor_initialized")

        sentiment_analyzer = SentimentAnalyzer(settings)
        await sentiment_analyzer.initialize()
        logger.info("sentiment_analyzer_initialized")

        response_generator = ResponseGenerator(settings)
        await response_generator.initialize()
        logger.info("response_generator_initialized")

        # Initialize main NLP engine
        nlp_engine = NLPEngine(
            text_processor=text_processor,
            intent_classifier=intent_classifier,
            entity_extractor=entity_extractor,
            sentiment_analyzer=sentiment_analyzer,
            response_generator=response_generator,
            embeddings_manager=embeddings_manager,
            cache_manager=cache_manager
        )
        await nlp_engine.initialize()
        logger.info("nlp_engine_initialized")

        # Start background tasks
        background_tasks = [
            asyncio.create_task(cache_cleanup_worker()),
            asyncio.create_task(model_health_check_worker()),
            asyncio.create_task(embeddings_sync_worker()),
        ]

        logger.info("background_workers_started", workers_count=len(background_tasks))

        try:
            yield
        finally:
            # Cleanup background tasks
            for task in background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Cleanup services
            if cache_manager:
                await cache_manager.cleanup()
            if nlp_engine:
                await nlp_engine.cleanup()
            if embeddings_manager:
                await embeddings_manager.cleanup()

            logger.info("ai_nlp_service_shutdown")

    except Exception as e:
        logger.error(
            "ai_nlp_service_startup_error",
            error=str(e),
            exc_info=True
        )
        raise


# Initialize FastAPI app
app = FastAPI(
    title="AI & NLP Service",
    description="Advanced Natural Language Processing service for intent recognition, entity extraction, and sentiment analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.middleware("http")(log_requests)

# Exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

app.include_router(
    intent.router,
    prefix="/api/v1/intent",
    tags=["Intent Recognition"]
)

app.include_router(
    entities.router,
    prefix="/api/v1/entities",
    tags=["Entity Extraction"]
)

app.include_router(
    sentiment.router,
    prefix="/api/v1/sentiment",
    tags=["Sentiment Analysis"]
)

app.include_router(
    language.router,
    prefix="/api/v1/language",
    tags=["Language Detection"]
)

app.include_router(
    response_gen.router,
    prefix="/api/v1/response",
    tags=["Response Generation"]
)

app.include_router(
    embeddings.router,
    prefix="/api/v1/embeddings",
    tags=["Embeddings & Similarity"]
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ai-nlp-service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": time.time(),
        "capabilities": [
            "intent_recognition",
            "entity_extraction",
            "sentiment_analysis",
            "language_detection",
            "response_generation",
            "text_embeddings"
        ]
    }


@app.get("/api/v1/status")
async def service_status():
    """Get comprehensive service status"""

    status_data = {
        "service": {
            "name": "ai-nlp-service",
            "version": "1.0.0",
            "status": "healthy",
            "environment": settings.environment,
            "timestamp": time.time(),
        },
        "components": {
            "nlp_engine": {
                "status": "ready" if nlp_engine and nlp_engine.is_ready() else "not_ready",
                "models_loaded": nlp_engine.get_loaded_models() if nlp_engine else [],
            },
            "text_processor": {
                "status": "active" if text_processor and text_processor.is_active() else "inactive",
            },
            "intent_classifier": {
                "status": "ready" if intent_classifier and intent_classifier.is_ready() else "not_ready",
                "available_intents": intent_classifier.get_available_intents() if intent_classifier else [],
            },
            "entity_extractor": {
                "status": "ready" if entity_extractor and entity_extractor.is_ready() else "not_ready",
                "supported_entities": entity_extractor.get_supported_entities() if entity_extractor else [],
            },
            "sentiment_analyzer": {
                "status": "ready" if sentiment_analyzer and sentiment_analyzer.is_ready() else "not_ready",
                "supported_languages": sentiment_analyzer.get_supported_languages() if sentiment_analyzer else [],
            },
            "embeddings_manager": {
                "status": "active" if embeddings_manager and embeddings_manager.is_active() else "inactive",
                "vector_db_connected": embeddings_manager.is_connected() if embeddings_manager else False,
            },
            "cache_manager": {
                "status": "active" if cache_manager and cache_manager.is_active() else "inactive",
                "cache_stats": await cache_manager.get_cache_stats() if cache_manager else {},
            }
        }
    }

    return status_data


# Comprehensive NLP processing endpoint
@app.post("/api/v1/process")
async def process_text(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Comprehensive text processing with all NLP capabilities"""

    start_time = time.time()

    try:
        text = request_data.get("text", "")
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text is required"
            )

        # Get processing options
        options = {
            "language": request_data.get("language", "auto"),
            "context": request_data.get("context", {}),
            "conversation_id": request_data.get("conversation_id"),
            "features": request_data.get("features", {
                "intent_recognition": True,
                "entity_extraction": True,
                "sentiment_analysis": True,
                "language_detection": True,
                "response_generation": False,
                "embeddings": False
            })
        }

        # Process with NLP engine
        result = await nlp_engine.process_text(
            text=text,
            options=options
        )

        processing_time = time.time() - start_time

        # Add processing metadata
        result["processing_metadata"] = {
            "processing_time_ms": processing_time * 1000,
            "service_version": "1.0.0",
            "timestamp": time.time(),
            "features_used": [k for k, v in options["features"].items() if v]
        }

        # Cache result if requested
        if request_data.get("cache_result", True):
            background_tasks.add_task(
                cache_result,
                text,
                result,
                options.get("cache_ttl", 3600)
            )

        logger.info(
            "text_processed_successfully",
            text_length=len(text),
            processing_time=processing_time,
            features_used=result["processing_metadata"]["features_used"]
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "text_processing_error",
            text_length=len(request_data.get("text", "")),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text processing failed"
        )


# Background workers
async def cache_cleanup_worker():
    """Background worker for cache cleanup"""
    logger.info("cache_cleanup_worker_started")

    while True:
        try:
            if cache_manager:
                # Clean expired entries every hour
                await cache_manager.cleanup_expired_entries()
                await asyncio.sleep(3600)  # 1 hour
            else:
                await asyncio.sleep(60)  # Wait and retry

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(
                "cache_cleanup_worker_error",
                error=str(e),
                exc_info=True
            )
            await asyncio.sleep(300)  # 5 minutes


async def model_health_check_worker():
    """Background worker for model health checking"""
    logger.info("model_health_check_worker_started")

    while True:
        try:
            if nlp_engine:
                # Check model health every 30 minutes
                health_status = await nlp_engine.health_check()

                if not health_status["healthy"]:
                    logger.warning(
                        "nlp_engine_health_check_failed",
                        issues=health_status.get("issues", [])
                    )

                await asyncio.sleep(1800)  # 30 minutes
            else:
                await asyncio.sleep(60)  # Wait and retry

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(
                "model_health_check_worker_error",
                error=str(e),
                exc_info=True
            )
            await asyncio.sleep(300)  # 5 minutes


async def embeddings_sync_worker():
    """Background worker for embeddings synchronization"""
    logger.info("embeddings_sync_worker_started")

    while True:
        try:
            if embeddings_manager:
                # Sync embeddings every 6 hours
                await embeddings_manager.sync_embeddings()
                await asyncio.sleep(21600)  # 6 hours
            else:
                await asyncio.sleep(300)  # Wait and retry

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(
                "embeddings_sync_worker_error",
                error=str(e),
                exc_info=True
            )
            await asyncio.sleep(1800)  # 30 minutes


async def cache_result(text: str, result: Dict[str, Any], ttl: int):
    """Cache processing result"""
    try:
        if cache_manager:
            cache_key = f"nlp_result:{hash(text)}"
            await cache_manager.set(cache_key, result, ttl)
    except Exception as e:
        logger.warning(
            "cache_result_error",
            error=str(e)
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
    )
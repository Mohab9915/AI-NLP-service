"""
NLP Engine - Main Natural Language Processing Engine
Orchestrates all NLP components and provides unified processing interface
"""
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import time
import json
from datetime import datetime

from shared.config.settings import get_settings
from shared.utils.logger import get_service_logger

settings = get_settings("ai-nlp-service")
logger = get_service_logger("nlp_engine")


class NLPEngine:
    """Main NLP processing engine"""

    def __init__(
        self,
        text_processor,
        intent_classifier,
        entity_extractor,
        sentiment_analyzer,
        response_generator,
        embeddings_manager,
        cache_manager
    ):
        self.text_processor = text_processor
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor
        self.sentiment_analyzer = sentiment_analyzer
        self.response_generator = response_generator
        self.embeddings_manager = embeddings_manager
        self.cache_manager = cache_manager

        self.loaded_models = {}
        self.is_initialized = False

    async def initialize(self):
        """Initialize all NLP components"""
        try:
            logger.info("initializing_nlp_engine")

            # Initialize all components
            await self.text_processor.initialize()
            await self.intent_classifier.initialize()
            await self.entity_extractor.initialize()
            await self.sentiment_analyzer.initialize()
            await self.response_generator.initialize()
            await self.embeddings_manager.initialize()

            # Load models
            await self._load_models()

            self.is_initialized = True
            logger.info("nlp_engine_initialized_successfully")

        except Exception as e:
            logger.error(
                "nlp_engine_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def cleanup(self):
        """Cleanup NLP components"""
        try:
            await self.text_processor.cleanup()
            await self.intent_classifier.cleanup()
            await self.entity_extractor.cleanup()
            await self.sentiment_analyzer.cleanup()
            await self.response_generator.cleanup()
            await self.embeddings_manager.cleanup()

            logger.info("nlp_engine_cleaned")

        except Exception as e:
            logger.error(
                "nlp_engine_cleanup_error",
                error=str(e),
                exc_info=True
            )

    def is_ready(self) -> bool:
        """Check if NLP engine is ready"""
        return (
            self.is_initialized and
            self.text_processor.is_active() and
            self.intent_classifier.is_ready() and
            self.entity_extractor.is_ready() and
            self.sentiment_analyzer.is_ready() and
            self.response_generator.is_ready()
        )

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded models"""
        return list(self.loaded_models.keys())

    async def process_text(
        self,
        text: str,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process text with specified NLP features"""

        if not self.is_ready():
            raise Exception("NLP engine not ready")

        start_time = time.time()

        # Default options
        default_options = {
            "language": "auto",
            "context": {},
            "conversation_id": None,
            "features": {
                "intent_recognition": True,
                "entity_extraction": True,
                "sentiment_analysis": True,
                "language_detection": True,
                "response_generation": False,
                "embeddings": False
            }
        }

        if options:
            default_options.update(options)

        options = default_options
        features = options["features"]

        result = {
            "text": text,
            "processed_at": time.time(),
            "results": {}
        }

        try:
            # Pre-process text
            processed_text = await self.text_processor.process_text(
                text,
                language=options.get("language")
            )
            result["processed_text"] = processed_text

            # Language detection
            if features.get("language_detection", True):
                language_result = await self._detect_language(processed_text)
                result["results"]["language"] = language_result

            # Intent recognition
            if features.get("intent_recognition", True):
                intent_result = await self._classify_intent(
                    processed_text,
                    context=options.get("context", {}),
                    language=result["results"]["language"].get("detected_language", "english")
                )
                result["results"]["intent"] = intent_result

            # Entity extraction
            if features.get("entity_extraction", True):
                entity_result = await self._extract_entities(
                    processed_text,
                    intent=result["results"].get("intent", {}),
                    language=result["results"]["language"].get("detected_language", "english")
                )
                result["results"]["entities"] = entity_result

            # Sentiment analysis
            if features.get("sentiment_analysis", True):
                sentiment_result = await self._analyze_sentiment(
                    processed_text,
                    language=result["results"]["language"].get("detected_language", "english")
                )
                result["results"]["sentiment"] = sentiment_result

            # Response generation
            if features.get("response_generation", False):
                response_result = await self._generate_response(
                    processed_text,
                    intent=result["results"].get("intent", {}),
                    entities=result["results"].get("entities", {}),
                    sentiment=result["results"].get("sentiment", {}),
                    context=options.get("context", {}),
                    conversation_id=options.get("conversation_id"),
                    language=result["results"]["language"].get("detected_language", "english")
                )
                result["results"]["response"] = response_result

            # Text embeddings
            if features.get("embeddings", False):
                embeddings_result = await self._generate_embeddings(
                    processed_text,
                    language=result["results"]["language"].get("detected_language", "english")
                )
                result["results"]["embeddings"] = embeddings_result

            processing_time = time.time() - start_time
            result["processing_time_ms"] = processing_time * 1000
            result["success"] = True

            logger.info(
                "text_processed_successfully",
                text_length=len(text),
                processing_time=processing_time,
                features_used=[k for k, v in features.items() if v]
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "text_processing_error",
                text_length=len(text),
                error=str(e),
                processing_time=processing_time,
                exc_info=True
            )

            result["processing_time_ms"] = processing_time * 1000
            result["success"] = False
            result["error"] = str(e)

        return result

    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of text"""
        try:
            # Check cache first
            cache_key = f"language_detection:{hash(text)}"
            cached_result = await self.cache_manager.get(cache_key)

            if cached_result:
                return cached_result

            # Detect language
            language_result = await self.text_processor.detect_language(text)

            # Cache result
            await self.cache_manager.set(cache_key, language_result, ttl=3600)

            return language_result

        except Exception as e:
            logger.error(
                "language_detection_error",
                error=str(e)
            )
            return {
                "detected_language": "english",
                "confidence": 0.5,
                "error": str(e)
            }

    async def _classify_intent(
        self,
        text: str,
        context: Dict[str, Any],
        language: str = "english"
    ) -> Dict[str, Any]:
        """Classify intent of text"""
        try:
            # Check cache first
            cache_key = f"intent_classification:{hash(text)}:{language}"
            cached_result = await self.cache_manager.get(cache_key)

            if cached_result:
                return cached_result

            # Classify intent
            intent_result = await self.intent_classifier.classify_intent(
                text=text,
                context=context,
                language=language
            )

            # Cache result
            await self.cache_manager.set(cache_key, intent_result, ttl=1800)

            return intent_result

        except Exception as e:
            logger.error(
                "intent_classification_error",
                error=str(e)
            )
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "alternative_intents": [],
                "error": str(e)
            }

    async def _extract_entities(
        self,
        text: str,
        intent: Dict[str, Any],
        language: str = "english"
    ) -> Dict[str, Any]:
        """Extract entities from text"""
        try:
            # Check cache first
            cache_key = f"entity_extraction:{hash(text)}:{language}"
            cached_result = await self.cache_manager.get(cache_key)

            if cached_result:
                return cached_result

            # Extract entities
            entity_result = await self.entity_extractor.extract_entities(
                text=text,
                intent=intent,
                language=language
            )

            # Cache result
            await self.cache_manager.set(cache_key, entity_result, ttl=1800)

            return entity_result

        except Exception as e:
            logger.error(
                "entity_extraction_error",
                error=str(e)
            )
            return {
                "entities": [],
                "relations": [],
                "error": str(e)
            }

    async def _analyze_sentiment(
        self,
        text: str,
        language: str = "english"
    ) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        try:
            # Check cache first
            cache_key = f"sentiment_analysis:{hash(text)}:{language}"
            cached_result = await self.cache_manager.get(cache_key)

            if cached_result:
                return cached_result

            # Analyze sentiment
            sentiment_result = await self.sentiment_analyzer.analyze_sentiment(
                text=text,
                language=language
            )

            # Cache result
            await self.cache_manager.set(cache_key, sentiment_result, ttl=1800)

            return sentiment_result

        except Exception as e:
            logger.error(
                "sentiment_analysis_error",
                error=str(e)
            )
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "emotion": None,
                "error": str(e)
            }

    async def _generate_response(
        self,
        text: str,
        intent: Dict[str, Any],
        entities: Dict[str, Any],
        sentiment: Dict[str, Any],
        context: Dict[str, Any],
        conversation_id: Optional[str],
        language: str = "english"
    ) -> Dict[str, Any]:
        """Generate response"""
        try:
            # Generate response
            response_result = await self.response_generator.generate_response(
                input_text=text,
                intent=intent,
                entities=entities,
                sentiment=sentiment,
                context=context,
                conversation_id=conversation_id,
                language=language
            )

            return response_result

        except Exception as e:
            logger.error(
                "response_generation_error",
                error=str(e)
            )
            return {
                "response": "I apologize, but I'm having trouble generating a response right now.",
                "confidence": 0.0,
                "error": str(e)
            }

    async def _generate_embeddings(
        self,
        text: str,
        language: str = "english"
    ) -> Dict[str, Any]:
        """Generate text embeddings"""
        try:
            # Check cache first
            cache_key = f"embeddings:{hash(text)}:{language}"
            cached_result = await self.cache_manager.get(cache_key)

            if cached_result:
                return cached_result

            # Generate embeddings
            embeddings_result = await self.embeddings_manager.generate_embeddings(
                text=text,
                language=language
            )

            # Cache result
            await self.cache_manager.set(cache_key, embeddings_result, ttl=7200)

            return embeddings_result

        except Exception as e:
            logger.error(
                "embeddings_generation_error",
                error=str(e)
            )
            return {
                "embeddings": [],
                "dimension": 0,
                "model": "unknown",
                "error": str(e)
            }

    async def _load_models(self):
        """Load required models"""
        try:
            logger.info("loading_nlp_models")

            # Load models using component-specific methods
            if hasattr(self.intent_classifier, 'load_model'):
                await self.intent_classifier.load_model()
                self.loaded_models["intent_classifier"] = True

            if hasattr(self.entity_extractor, 'load_model'):
                await self.entity_extractor.load_model()
                self.loaded_models["entity_extractor"] = True

            if hasattr(self.sentiment_analyzer, 'load_model'):
                await self.sentiment_analyzer.load_model()
                self.loaded_models["sentiment_analyzer"] = True

            if hasattr(self.embeddings_manager, 'load_model'):
                await self.embeddings_manager.load_model()
                self.loaded_models["embeddings_manager"] = True

            logger.info(
                "nlp_models_loaded",
                loaded_models=list(self.loaded_models.keys())
            )

        except Exception as e:
            logger.error(
                "model_loading_error",
                error=str(e),
                exc_info=True
            )
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health_status = {
            "healthy": True,
            "components": {},
            "issues": []
        }

        try:
            # Check each component
            components = [
                ("text_processor", self.text_processor),
                ("intent_classifier", self.intent_classifier),
                ("entity_extractor", self.entity_extractor),
                ("sentiment_analyzer", self.sentiment_analyzer),
                ("response_generator", self.response_generator),
                ("embeddings_manager", self.embeddings_manager),
                ("cache_manager", self.cache_manager),
            ]

            for component_name, component in components:
                try:
                    if hasattr(component, 'health_check'):
                        component_health = await component.health_check()
                        health_status["components"][component_name] = component_health

                        if not component_health.get("healthy", True):
                            health_status["healthy"] = False
                            health_status["issues"].append(f"{component_name}: {component_health.get('issue', 'Unknown issue')}")
                    else:
                        # Simple check - if component has is_ready or is_active method
                        if hasattr(component, 'is_ready'):
                            is_healthy = component.is_ready()
                        elif hasattr(component, 'is_active'):
                            is_healthy = component.is_active()
                        else:
                            is_healthy = True  # Assume healthy if no check method

                        health_status["components"][component_name] = {
                            "healthy": is_healthy,
                            "status": "active" if is_healthy else "inactive"
                        }

                        if not is_healthy:
                            health_status["healthy"] = False
                            health_status["issues"].append(f"{component_name}: Component is not healthy")

                except Exception as e:
                    health_status["components"][component_name] = {
                        "healthy": False,
                        "error": str(e)
                    }
                    health_status["healthy"] = False
                    health_status["issues"].append(f"{component_name}: {str(e)}")

            return health_status

        except Exception as e:
            logger.error(
                "nlp_engine_health_check_error",
                error=str(e),
                exc_info=True
            )
            return {
                "healthy": False,
                "components": {},
                "issues": [f"Health check failed: {str(e)}"]
            }

    async def batch_process(
        self,
        texts: List[str],
        options: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple texts in batch"""
        try:
            # Process texts concurrently
            tasks = [
                self.process_text(text, options)
                for text in texts
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "text": texts[i],
                        "success": False,
                        "error": str(result),
                        "processed_at": time.time()
                    })
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error(
                "batch_processing_error",
                texts_count=len(texts),
                error=str(e),
                exc_info=True
            )
            raise
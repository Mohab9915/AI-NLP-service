"""
Embeddings Manager - Advanced Text Embeddings and Vector Similarity
Manages text embeddings generation, storage, and similarity search for semantic understanding
"""
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import time
import json
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from shared.config.settings import get_settings
from shared.utils.logger import get_service_logger

settings = get_settings("ai-nlp-service")
logger = get_service_logger("embeddings_manager")


class EmbeddingModel(Enum):
    """Supported embedding models"""
    SENTENCE_TRANSFORMER = "sentence-transformer"
    OPENAI_ADA = "openai-ada"
    DISTILBERT = "distilbert"
    MULTILINGUAL = "multilingual"


class VectorDistance(Enum):
    """Distance metrics for similarity"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"


@dataclass
class TextEmbedding:
    """Text embedding with metadata"""
    text: str
    embedding: List[float]
    model: str
    dimension: int
    language: str
    created_at: float
    text_hash: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarityResult:
    """Similarity search result"""
    text: str
    similarity_score: float
    embedding_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance_type: str = "cosine"


class EmbeddingCache:
    """In-memory cache for embeddings"""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, TextEmbedding] = {}
        self.timestamps: Dict[str, float] = {}
        self.access_times: Dict[str, float] = {}

    def get(self, key: str) -> Optional[TextEmbedding]:
        """Get embedding from cache"""
        if key not in self.cache:
            return None

        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            self.remove(key)
            return None

        # Update access time
        self.access_times[key] = time.time()
        return self.cache[key]

    def put(self, key: str, embedding: TextEmbedding):
        """Put embedding in cache"""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self.remove(oldest_key)

        self.cache[key] = embedding
        self.timestamps[key] = time.time()
        self.access_times[key] = time.time()

    def remove(self, key: str):
        """Remove embedding from cache"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_times.pop(key, None)

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.timestamps.clear()
        self.access_times.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
            "ttl_seconds": self.ttl_seconds
        }


class MockVectorDB:
    """Mock vector database for demonstration (in production, use real vector DB like Pinecone, Weaviate, or Chroma)"""

    def __init__(self):
        self.embeddings: Dict[str, TextEmbedding] = {}
        self.index: Dict[str, List[str]] = {}  # Language-based index

    def add_embedding(self, embedding: TextEmbedding) -> str:
        """Add embedding to database"""
        embedding_id = str(uuid.uuid4())
        embedding.metadata["embedding_id"] = embedding_id
        self.embeddings[embedding_id] = embedding

        # Add to language index
        lang = embedding.language
        if lang not in self.index:
            self.index[lang] = []
        self.index[lang].append(embedding_id)

        return embedding_id

    def get_embedding(self, embedding_id: str) -> Optional[TextEmbedding]:
        """Get embedding by ID"""
        return self.embeddings.get(embedding_id)

    def search_similar(
        self,
        query_embedding: List[float],
        language: str = "english",
        top_k: int = 10,
        distance_metric: VectorDistance = VectorDistance.COSINE,
        threshold: float = 0.5
    ) -> List[SimilarityResult]:
        """Search for similar embeddings"""

        if language not in self.index:
            return []

        results = []
        query_vector = np.array(query_embedding)

        for embedding_id in self.index[language]:
            stored_embedding = self.embeddings[embedding_id]
            stored_vector = np.array(stored_embedding.embedding)

            # Calculate similarity
            similarity = self._calculate_similarity(
                query_vector,
                stored_vector,
                distance_metric
            )

            if similarity >= threshold:
                results.append(SimilarityResult(
                    text=stored_embedding.text,
                    similarity_score=similarity,
                    embedding_id=embedding_id,
                    metadata=stored_embedding.metadata,
                    distance_type=distance_metric.value
                ))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]

    def _calculate_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: VectorDistance
    ) -> float:
        """Calculate similarity between vectors"""

        if metric == VectorDistance.COSINE:
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

        elif metric == VectorDistance.EUCLIDEAN:
            # Convert Euclidean distance to similarity
            distance = np.linalg.norm(vec1 - vec2)
            return 1 / (1 + distance)  # Convert distance to similarity

        elif metric == VectorDistance.DOT_PRODUCT:
            # Dot product similarity
            return np.dot(vec1, vec2)

        elif metric == VectorDistance.MANHATTAN:
            # Convert Manhattan distance to similarity
            distance = np.sum(np.abs(vec1 - vec2))
            return 1 / (1 + distance)

        return 0.0

    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete embedding from database"""
        if embedding_id not in self.embeddings:
            return False

        embedding = self.embeddings[embedding_id]
        language = embedding.language

        # Remove from main storage
        del self.embeddings[embedding_id]

        # Remove from language index
        if language in self.index:
            self.index[language].remove(embedding_id)
            if not self.index[language]:
                del self.index[language]

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_embeddings": len(self.embeddings),
            "languages": {lang: len(ids) for lang, ids in self.index.items()},
            "dimension": next(iter(self.embeddings.values())).dimension if self.embeddings else 0
        }


class EmbeddingsManager:
    """Advanced embeddings management system"""

    def __init__(self, settings):
        self.settings = settings
        self.cache = EmbeddingCache(
            max_size=getattr(settings, 'embedding_cache_size', 10000),
            ttl_seconds=getattr(settings, 'embedding_cache_ttl', 3600)
        )
        self.vector_db = MockVectorDB()
        self.is_initialized = False

        # Embedding model configuration
        self.models = {
            EmbeddingModel.SENTENCE_TRANSFORMER: {
                "dimension": 384,
                "languages": ["english", "arabic", "hebrew"],
                "max_length": 512
            },
            EmbeddingModel.MULTILINGUAL: {
                "dimension": 768,
                "languages": ["english", "arabic", "hebrew", "spanish", "french", "german"],
                "max_length": 512
            }
        }

        self.active_model = EmbeddingModel.SENTENCE_TRANSFORMER

    async def initialize(self):
        """Initialize the embeddings manager"""
        try:
            logger.info("initializing_embeddings_manager")

            # Initialize embedding models (mock implementation)
            await self._load_embedding_models()

            # Load pre-trained embeddings if available
            await self._load_pretrained_embeddings()

            self.is_initialized = True
            logger.info("embeddings_manager_initialized")

        except Exception as e:
            logger.error(
                "embeddings_manager_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def cleanup(self):
        """Cleanup embeddings manager resources"""
        try:
            self.cache.clear()
            self.vector_db.embeddings.clear()
            self.vector_db.index.clear()
            logger.info("embeddings_manager_cleaned")

        except Exception as e:
            logger.error(
                "embeddings_manager_cleanup_error",
                error=str(e)
            )

    def is_ready(self) -> bool:
        """Check if embeddings manager is ready"""
        return self.is_initialized

    def is_connected(self) -> bool:
        """Check if connected to vector database"""
        return True  # Mock implementation always returns True

    async def generate_embeddings(
        self,
        text: str,
        language: str = "english",
        model: Optional[EmbeddingModel] = None,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Generate text embeddings"""

        start_time = time.time()

        try:
            # Use default model if not specified
            if not model:
                model = self.active_model

            # Check cache first
            text_hash = self._hash_text(text)
            cache_key = f"{model.value}:{language}:{text_hash}"
            cached_embedding = self.cache.get(cache_key)

            if cached_embedding:
                processing_time = time.time() - start_time
                logger.info(
                    "embeddings_retrieved_from_cache",
                    text_length=len(text),
                    processing_time=processing_time
                )

                return {
                    "embeddings": cached_embedding.embedding,
                    "dimension": cached_embedding.dimension,
                    "model": cached_embedding.model,
                    "language": cached_embedding.language,
                    "cached": True,
                    "processing_time_ms": processing_time * 1000
                }

            # Generate new embeddings
            embeddings = await self._generate_embeddings_internal(
                text,
                language,
                model
            )

            # Create embedding object
            embedding_obj = TextEmbedding(
                text=text,
                embedding=embeddings,
                model=model.value,
                dimension=len(embeddings),
                language=language,
                created_at=time.time(),
                text_hash=text_hash,
                metadata={"generated_by": "embeddings_manager"}
            )

            # Cache the result
            self.cache.put(cache_key, embedding_obj)

            # Store in vector database
            embedding_id = self.vector_db.add_embedding(embedding_obj)

            processing_time = time.time() - start_time

            result = {
                "embeddings": embeddings,
                "dimension": len(embeddings),
                "model": model.value,
                "language": language,
                "cached": False,
                "embedding_id": embedding_id,
                "processing_time_ms": processing_time * 1000,
                "text_hash": text_hash
            }

            logger.info(
                "embeddings_generated_successfully",
                text_length=len(text),
                model=model.value,
                language=language,
                dimension=len(embeddings),
                processing_time=processing_time
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "embeddings_generation_error",
                text_length=len(text),
                language=language,
                error=str(e),
                processing_time=processing_time,
                exc_info=True
            )

            return {
                "embeddings": [],
                "dimension": 0,
                "model": model.value if model else "unknown",
                "language": language,
                "cached": False,
                "error": str(e),
                "processing_time_ms": processing_time * 1000
            }

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        language: str = "english",
        model: Optional[EmbeddingModel] = None,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for multiple texts in batch"""

        try:
            if not model:
                model = self.active_model

            # Process texts in batches
            all_results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = await self._process_batch(batch, language, model)
                all_results.extend(batch_results)

            logger.info(
                "batch_embeddings_generated",
                total_texts=len(texts),
                model=model.value,
                language=language
            )

            return all_results

        except Exception as e:
            logger.error(
                "batch_embeddings_generation_error",
                texts_count=len(texts),
                error=str(e),
                exc_info=True
            )
            raise

    async def search_similar_texts(
        self,
        query_text: str,
        language: str = "english",
        top_k: int = 10,
        threshold: float = 0.5,
        distance_metric: VectorDistance = VectorDistance.COSINE,
        include_embeddings: bool = False
    ) -> Dict[str, Any]:
        """Search for similar texts using embeddings"""

        start_time = time.time()

        try:
            # Generate query embeddings
            query_result = await self.generate_embeddings(
                query_text,
                language=language
            )

            if not query_result.get("embeddings"):
                return {
                    "query_text": query_text,
                    "similar_texts": [],
                    "total_found": 0,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "error": "Failed to generate query embeddings"
                }

            query_embeddings = query_result["embeddings"]

            # Search for similar embeddings
            similar_results = self.vector_db.search_similar(
                query_embeddings,
                language=language,
                top_k=top_k,
                distance_metric=distance_metric,
                threshold=threshold
            )

            # Format results
            formatted_results = []
            for result in similar_results:
                result_data = {
                    "text": result.text,
                    "similarity_score": result.similarity_score,
                    "embedding_id": result.embedding_id,
                    "distance_type": result.distance_type,
                    "metadata": result.metadata
                }

                if include_embeddings:
                    embedding = self.vector_db.get_embedding(result.embedding_id)
                    if embedding:
                        result_data["embeddings"] = embedding.embedding

                formatted_results.append(result_data)

            processing_time = time.time() - start_time

            return {
                "query_text": query_text,
                "similar_texts": formatted_results,
                "total_found": len(formatted_results),
                "threshold": threshold,
                "distance_metric": distance_metric.value,
                "language": language,
                "processing_time_ms": processing_time * 1000,
                "query_embedding_id": query_result.get("embedding_id")
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "similar_texts_search_error",
                query_text_length=len(query_text),
                error=str(e),
                processing_time=processing_time,
                exc_info=True
            )

            return {
                "query_text": query_text,
                "similar_texts": [],
                "total_found": 0,
                "processing_time_ms": processing_time * 1000,
                "error": str(e)
            }

    async def calculate_similarity(
        self,
        text1: str,
        text2: str,
        language: str = "english",
        distance_metric: VectorDistance = VectorDistance.COSINE
    ) -> Dict[str, Any]:
        """Calculate similarity between two texts"""

        start_time = time.time()

        try:
            # Generate embeddings for both texts
            embeddings1_result = await self.generate_embeddings(text1, language)
            embeddings2_result = await self.generate_embeddings(text2, language)

            if (not embeddings1_result.get("embeddings") or
                not embeddings2_result.get("embeddings")):
                return {
                    "text1": text1,
                    "text2": text2,
                    "similarity_score": 0.0,
                    "error": "Failed to generate embeddings"
                }

            # Calculate similarity
            vec1 = np.array(embeddings1_result["embeddings"])
            vec2 = np.array(embeddings2_result["embeddings"])

            similarity_score = self.vector_db._calculate_similarity(
                vec1, vec2, distance_metric
            )

            processing_time = time.time() - start_time

            return {
                "text1": text1,
                "text2": text2,
                "similarity_score": float(similarity_score),
                "distance_metric": distance_metric.value,
                "language": language,
                "processing_time_ms": processing_time * 1000,
                "embedding1_id": embeddings1_result.get("embedding_id"),
                "embedding2_id": embeddings2_result.get("embedding_id")
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "similarity_calculation_error",
                text1_length=len(text1),
                text2_length=len(text2),
                error=str(e),
                processing_time=processing_time
            )

            return {
                "text1": text1,
                "text2": text2,
                "similarity_score": 0.0,
                "error": str(e),
                "processing_time_ms": processing_time * 1000
            }

    async def delete_embedding(self, embedding_id: str) -> Dict[str, Any]:
        """Delete embedding from database"""

        try:
            success = self.vector_db.delete_embedding(embedding_id)

            if success:
                logger.info(
                    "embedding_deleted_successfully",
                    embedding_id=embedding_id
                )
                return {"deleted": True, "embedding_id": embedding_id}
            else:
                logger.warning(
                    "embedding_not_found",
                    embedding_id=embedding_id
                )
                return {"deleted": False, "embedding_id": embedding_id, "error": "Embedding not found"}

        except Exception as e:
            logger.error(
                "embedding_deletion_error",
                embedding_id=embedding_id,
                error=str(e)
            )
            return {"deleted": False, "embedding_id": embedding_id, "error": str(e)}

    async def sync_embeddings(self):
        """Synchronize embeddings (placeholder for future implementation)"""

        try:
            logger.info("syncing_embeddings")
            # In a real implementation, this would sync with external vector DB
            # For now, just log the sync attempt
            await asyncio.sleep(1)  # Simulate sync operation
            logger.info("embeddings_synced_successfully")

        except Exception as e:
            logger.error(
                "embeddings_sync_error",
                error=str(e)
            )

    async def get_embeddings_stats(self) -> Dict[str, Any]:
        """Get embeddings statistics"""

        try:
            db_stats = self.vector_db.get_stats()
            cache_stats = self.cache.stats()

            return {
                "vector_database": db_stats,
                "cache": cache_stats,
                "active_model": self.active_model.value,
                "available_models": [model.value for model in self.models.keys()],
                "supported_languages": list(set(lang for model in self.models.values() for lang in model["languages"]))
            }

        except Exception as e:
            logger.error(
                "embeddings_stats_error",
                error=str(e)
            )
            return {}

    async def _load_embedding_models(self):
        """Load embedding models (mock implementation)"""
        try:
            logger.info("loading_embedding_models")
            # In a real implementation, this would load actual ML models
            # For now, we simulate model loading
            await asyncio.sleep(0.5)
            logger.info("embedding_models_loaded")

        except Exception as e:
            logger.error(
                "embedding_models_loading_error",
                error=str(e)
            )
            raise

    async def _load_pretrained_embeddings(self):
        """Load pretrained embeddings (placeholder)"""
        try:
            logger.info("loading_pretrained_embeddings")
            # In a real implementation, this would load pretrained embeddings
            # For now, we simulate loading some common embeddings
            await asyncio.sleep(0.5)
            logger.info("pretrained_embeddings_loaded")

        except Exception as e:
            logger.error(
                "pretrained_embeddings_loading_error",
                error=str(e)
            )

    async def _generate_embeddings_internal(
        self,
        text: str,
        language: str,
        model: EmbeddingModel
    ) -> List[float]:
        """Generate embeddings using internal model (mock implementation)"""

        try:
            model_config = self.models[model]

            # Validate language support
            if language not in model_config["languages"]:
                logger.warning(
                    "language_not_supported_by_model",
                    language=language,
                    model=model.value,
                    falling_back_to="english"
                )
                language = "english"

            # Mock embedding generation
            # In a real implementation, this would use actual ML models
            dimension = model_config["dimension"]

            # Generate pseudo-random but deterministic embeddings based on text
            text_hash = hash(text + language + model.value)
            np.random.seed(text_hash % (2**32))
            embedding = np.random.normal(0, 1, dimension).tolist()

            # Normalize the embedding
            embedding_norm = np.array(embedding)
            embedding_norm = embedding_norm / np.linalg.norm(embedding_norm)

            return embedding_norm.tolist()

        except Exception as e:
            logger.error(
                "internal_embedding_generation_error",
                error=str(e)
            )
            raise

    async def _process_batch(
        self,
        texts: List[str],
        language: str,
        model: EmbeddingModel
    ) -> List[Dict[str, Any]]:
        """Process a batch of texts for embedding generation"""

        batch_results = []
        for text in texts:
            result = await self.generate_embeddings(text, language, model)
            batch_results.append(result)

        return batch_results

    def _hash_text(self, text: str) -> str:
        """Generate hash for text"""
        return str(hash(text))
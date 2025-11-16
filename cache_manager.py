"""
Cache Manager - High-Performance Multi-Level Caching System
Provides intelligent caching with Redis backend and local memory cache for optimal performance
"""
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
import time
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from shared.config.settings import get_settings
from shared.utils.logger import get_service_logger

settings = get_settings("ai-nlp-service")
logger = get_service_logger("cache_manager")


class CacheLevel(Enum):
    """Cache levels"""
    MEMORY = "memory"
    REDIS = "redis"
    DISTRIBUTED = "distributed"


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    FIFO = "first_in_first_out"
    TTL = "time_to_live"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_level: CacheLevel = CacheLevel.MEMORY

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def update_access(self):
        """Update access information"""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache:
    """In-memory LRU cache with TTL support"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU eviction
        self.total_size = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check expiration
        if entry.is_expired():
            self._remove_entry(key)
            return None

        # Update access information
        entry.update_access()
        self._update_access_order(key)

        return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache"""
        try:
            # Calculate size
            if ttl is None:
                ttl = self.default_ttl

            serialized_value = self._serialize_value(value)
            size_bytes = len(serialized_value)

            # Check if we need to evict entries
            while (len(self.cache) >= self.max_size or
                   self.total_size + size_bytes > self.max_size * 1024):  # Assume max_size is in MB
                if not self._evict_lru():
                    break

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
                size_bytes=size_bytes,
                cache_level=CacheLevel.MEMORY
            )

            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)

            # Add new entry
            self.cache[key] = entry
            self._update_access_order(key)
            self.total_size += size_bytes

            return True

        except Exception as e:
            logger.error(
                "memory_cache_put_error",
                key=key,
                error=str(e)
            )
            return False

    def remove(self, key: str) -> bool:
        """Remove entry from cache"""
        return self._remove_entry(key)

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
        self.total_size = 0

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count"""
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_entry(key)

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "entries_count": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
            "total_size_bytes": self.total_size,
            "average_entry_size": self.total_size / len(self.cache) if self.cache else 0,
            "hit_rate": getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
        }

    def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache"""
        if key not in self.cache:
            return False

        entry = self.cache[key]
        self.total_size -= entry.size_bytes
        del self.cache[key]

        # Remove from access order
        if key in self.access_order:
            self.access_order.remove(key)

        return True

    def _update_access_order(self, key: str):
        """Update LRU access order"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def _evict_lru(self) -> bool:
        """Evict least recently used entry"""
        if not self.access_order:
            return False

        lru_key = self.access_order.pop(0)
        self._remove_entry(lru_key)
        return True

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for size calculation"""
        try:
            return pickle.dumps(value)
        except Exception:
            # Fallback to JSON serialization
            return json.dumps(value, default=str).encode('utf-8')


class MockRedisClient:
    """Mock Redis client for demonstration (in production, use actual redis-py)"""

    def __init__(self):
        self.data: Dict[str, Dict[str, Any]] = {}

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        if key not in self.data:
            return None

        entry = self.data[key]

        # Check TTL
        if entry.get("expires_at") and time.time() > entry["expires_at"]:
            del self.data[key]
            return None

        return entry["value"]

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiration"""
        try:
            entry = {
                "value": value,
                "created_at": time.time()
            }

            if ex:
                entry["expires_at"] = time.time() + ex

            self.data[key] = entry
            return True

        except Exception as e:
            logger.error(
                "mock_redis_set_error",
                key=key,
                error=str(e)
            )
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if key in self.data:
            del self.data[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if key not in self.data:
            return False

        entry = self.data[key]
        if entry.get("expires_at") and time.time() > entry["expires_at"]:
            del self.data[key]
            return False

        return True

    async def flushdb(self) -> bool:
        """Clear all keys from Redis"""
        self.data.clear()
        return True

    async def info(self) -> Dict[str, Any]:
        """Get Redis information"""
        return {
            "used_memory": len(str(self.data)),
            "used_memory_human": f"{len(str(self.data))}B",
            "keyspace_hits": getattr(self, "_hits", 0),
            "keyspace_misses": getattr(self, "_misses", 0)
        }


class CacheManager:
    """Advanced multi-level cache manager"""

    def __init__(self, settings):
        self.settings = settings
        self.memory_cache = MemoryCache(
            max_size=getattr(settings, 'cache_memory_size', 1000),
            default_ttl=getattr(settings, 'cache_default_ttl', 3600)
        )
        self.redis_client = MockRedisClient()  # In production, use actual Redis client
        self.is_initialized = False

        # Cache configuration
        self.config = {
            "enable_memory_cache": True,
            "enable_redis_cache": True,
            "default_ttl": getattr(settings, 'cache_default_ttl', 3600),
            "max_memory_size": getattr(settings, 'cache_memory_size', 1000),
            "cleanup_interval": getattr(settings, 'cache_cleanup_interval', 300),
            "compression_threshold": 1024,  # Compress values larger than 1KB
            "enable_serialization": True
        }

        # Statistics
        self.stats = {
            "memory_hits": 0,
            "redis_hits": 0,
            "misses": 0,
            "puts": 0,
            "evictions": 0,
            "errors": 0
        }

    async def initialize(self):
        """Initialize the cache manager"""
        try:
            logger.info("initializing_cache_manager")

            # Initialize Redis connection (mock)
            if self.config["enable_redis_cache"]:
                await self._initialize_redis()

            # Start background cleanup task
            asyncio.create_task(self._background_cleanup())

            self.is_initialized = True
            logger.info("cache_manager_initialized")

        except Exception as e:
            logger.error(
                "cache_manager_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def cleanup(self):
        """Cleanup cache manager resources"""
        try:
            self.memory_cache.clear()
            if self.redis_client:
                await self.redis_client.flushdb()
            logger.info("cache_manager_cleaned")

        except Exception as e:
            logger.error(
                "cache_manager_cleanup_error",
                error=str(e)
            )

    def is_active(self) -> bool:
        """Check if cache manager is active"""
        return self.is_initialized

    async def get(self, key: str, use_memory: bool = True, use_redis: bool = True) -> Optional[Any]:
        """Get value from cache with multi-level fallback"""

        try:
            # Try memory cache first (fastest)
            if use_memory and self.config["enable_memory_cache"]:
                value = self.memory_cache.get(key)
                if value is not None:
                    self.stats["memory_hits"] += 1
                    logger.debug(
                        "cache_hit_memory",
                        key=key
                    )
                    return value

            # Try Redis cache
            if use_redis and self.config["enable_redis_cache"]:
                serialized_value = await self.redis_client.get(key)
                if serialized_value:
                    value = self._deserialize_value(serialized_value)

                    # Cache in memory for faster future access
                    if use_memory and self.config["enable_memory_cache"]:
                        self.memory_cache.put(key, value, ttl=self.config["default_ttl"])

                    self.stats["redis_hits"] += 1
                    logger.debug(
                        "cache_hit_redis",
                        key=key
                    )
                    return value

            # Cache miss
            self.stats["misses"] += 1
            logger.debug(
                "cache_miss",
                key=key
            )
            return None

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(
                "cache_get_error",
                key=key,
                error=str(e)
            )
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        use_memory: bool = True,
        use_redis: bool = True,
        compress: Optional[bool] = None
    ) -> bool:
        """Set value in cache with multi-level storage"""

        try:
            if ttl is None:
                ttl = self.config["default_ttl"]

            # Determine compression
            if compress is None:
                compress = self._should_compress(value)

            # Serialize value
            if compress:
                serialized_value = self._compress_and_serialize(value)
            else:
                serialized_value = self._serialize_value(value)

            # Store in memory cache
            if use_memory and self.config["enable_memory_cache"]:
                self.memory_cache.put(key, value, ttl=ttl)

            # Store in Redis cache
            if use_redis and self.config["enable_redis_cache"]:
                success = await self.redis_client.set(
                    key,
                    serialized_value,
                    ex=int(ttl) if ttl > 0 else None
                )
                if not success:
                    logger.warning(
                        "redis_cache_set_failed",
                        key=key
                    )

            self.stats["puts"] += 1
            logger.debug(
                "cache_set",
                key=key,
                ttl=ttl,
                compressed=compress
            )

            return True

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(
                "cache_set_error",
                key=key,
                error=str(e)
            )
            return False

    async def delete(self, key: str, use_memory: bool = True, use_redis: bool = True) -> bool:
        """Delete key from all cache levels"""

        try:
            success = True

            # Delete from memory cache
            if use_memory and self.config["enable_memory_cache"]:
                if not self.memory_cache.remove(key):
                    success = False

            # Delete from Redis cache
            if use_redis and self.config["enable_redis_cache"]:
                if not await self.redis_client.delete(key):
                    success = False

            logger.debug(
                "cache_delete",
                key=key,
                success=success
            )

            return success

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(
                "cache_delete_error",
                key=key,
                error=str(e)
            )
            return False

    async def exists(self, key: str, use_memory: bool = True, use_redis: bool = True) -> bool:
        """Check if key exists in cache"""

        try:
            # Check memory cache first
            if use_memory and self.config["enable_memory_cache"]:
                if self.memory_cache.get(key) is not None:
                    return True

            # Check Redis cache
            if use_redis and self.config["enable_redis_cache"]:
                if await self.redis_client.exists(key):
                    return True

            return False

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(
                "cache_exists_error",
                key=key,
                error=str(e)
            )
            return False

    async def get_or_set(
        self,
        key: str,
        value_func: Callable,
        ttl: Optional[float] = None,
        use_memory: bool = True,
        use_redis: bool = True
    ) -> Any:
        """Get value from cache or compute and set it"""

        # Try to get from cache first
        cached_value = await self.get(key, use_memory, use_redis)
        if cached_value is not None:
            return cached_value

        # Compute value
        try:
            if asyncio.iscoroutinefunction(value_func):
                value = await value_func()
            else:
                value = value_func()

            # Set in cache
            await self.set(key, value, ttl, use_memory, use_redis)
            return value

        except Exception as e:
            logger.error(
                "cache_get_or_set_error",
                key=key,
                error=str(e)
            )
            raise

    async def cleanup_expired_entries(self) -> Dict[str, int]:
        """Clean up expired entries from all cache levels"""

        try:
            cleanup_stats = {
                "memory": 0,
                "redis": 0
            }

            # Clean memory cache
            cleanup_stats["memory"] = self.memory_cache.cleanup_expired()

            # Clean Redis cache (mock implementation)
            # In real Redis, expired keys are cleaned automatically
            cleanup_stats["redis"] = 0  # Redis handles this automatically

            logger.info(
                "cache_cleanup_completed",
                cleanup_stats=cleanup_stats
            )

            return cleanup_stats

        except Exception as e:
            logger.error(
                "cache_cleanup_error",
                error=str(e)
            )
            return {"memory": 0, "redis": 0}

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""

        try:
            memory_stats = self.memory_cache.get_stats()
            redis_stats = await self.redis_client.info() if self.redis_client else {}

            # Calculate hit rates
            total_requests = (
                self.stats["memory_hits"] +
                self.stats["redis_hits"] +
                self.stats["misses"]
            )

            total_hits = self.stats["memory_hits"] + self.stats["redis_hits"]

            return {
                "overall": {
                    "total_requests": total_requests,
                    "total_hits": total_hits,
                    "total_misses": self.stats["misses"],
                    "hit_rate": total_hits / max(total_requests, 1),
                    "puts": self.stats["puts"],
                    "evictions": self.stats["evictions"],
                    "errors": self.stats["errors"]
                },
                "memory_cache": memory_stats,
                "redis_cache": redis_stats,
                "config": self.config
            }

        except Exception as e:
            logger.error(
                "cache_stats_error",
                error=str(e)
            )
            return {}

    async def clear_all(self) -> bool:
        """Clear all cache entries"""

        try:
            # Clear memory cache
            self.memory_cache.clear()

            # Clear Redis cache
            if self.redis_client:
                await self.redis_client.flushdb()

            # Reset statistics
            for key in self.stats:
                self.stats[key] = 0

            logger.info("cache_cleared_all_levels")
            return True

        except Exception as e:
            logger.error(
                "cache_clear_error",
                error=str(e)
            )
            return False

    def generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from components"""

        # Create a deterministic key
        components = [prefix] + [str(arg) for arg in args]

        # Add sorted kwargs
        for k, v in sorted(kwargs.items()):
            components.append(f"{k}:{v}")

        key_string = ":".join(components)

        # Hash long keys to avoid issues with very long cache keys
        if len(key_string) > 200:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"{prefix}:{key_hash}"

        return key_string

    async def _initialize_redis(self):
        """Initialize Redis connection (mock)"""
        try:
            logger.info("initializing_redis_connection")
            # In production, this would create actual Redis connection
            await asyncio.sleep(0.1)  # Simulate connection time
            logger.info("redis_connection_initialized")

        except Exception as e:
            logger.error(
                "redis_initialization_error",
                error=str(e)
            )
            raise

    async def _background_cleanup(self):
        """Background task for periodic cleanup"""
        while True:
            try:
                await asyncio.sleep(self.config["cleanup_interval"])
                await self.cleanup_expired_entries()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "background_cleanup_error",
                    error=str(e)
                )

    def _serialize_value(self, value: Any) -> str:
        """Serialize value for storage"""
        try:
            if self.config["enable_serialization"]:
                return json.dumps(value, default=str)
            else:
                return str(value)

        except Exception as e:
            logger.error(
                "value_serialization_error",
                error=str(e)
            )
            return str(value)

    def _deserialize_value(self, serialized_value: str) -> Any:
        """Deserialize value from storage"""
        try:
            if self.config["enable_serialization"]:
                return json.loads(serialized_value)
            else:
                return serialized_value

        except Exception as e:
            logger.error(
                "value_deserialization_error",
                error=str(e)
            )
            return serialized_value

    def _compress_and_serialize(self, value: Any) -> str:
        """Compress and serialize value (mock implementation)"""
        # In a real implementation, this would use compression libraries
        serialized = self._serialize_value(value)
        return f"COMPRESSED:{serialized}"

    def _decompress_and_deserialize(self, compressed_value: str) -> Any:
        """Decompress and deserialize value (mock implementation)"""
        # In a real implementation, this would use compression libraries
        if compressed_value.startswith("COMPRESSED:"):
            serialized = compressed_value[11:]  # Remove "COMPRESSED:" prefix
            return self._deserialize_value(serialized)
        return self._deserialize_value(compressed_value)

    def _should_compress(self, value: Any) -> bool:
        """Determine if value should be compressed"""
        try:
            serialized = self._serialize_value(value)
            return len(serialized) > self.config["compression_threshold"]
        except Exception:
            return False
"""
Health Check Router
API endpoints for health monitoring and system status
"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List
import time
import psutil
import asyncio

from shared.utils.logger import get_service_logger

router = APIRouter()
logger = get_service_logger("health_router")


# Global variables (will be injected by main app)
nlp_engine = None
text_processor = None
intent_classifier = None
entity_extractor = None
sentiment_analyzer = None
response_generator = None
embeddings_manager = None
cache_manager = None


async def get_nlp_engine():
    """Dependency to get NLP engine instance"""
    global nlp_engine
    return nlp_engine


@router.get("/", summary="Overall Service Health")
async def overall_health():
    """
    Get overall health status of the AI & NLP service.
    """
    try:
        health_status = {
            "service": "ai-nlp-service",
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - _get_service_start_time(),
            "version": "1.0.0",
            "components": {},
            "system": _get_system_info()
        }

        # Check individual components
        component_checks = [
            ("nlp_engine", nlp_engine, lambda x: x.is_ready() if x else False),
            ("text_processor", text_processor, lambda x: x.is_active() if x else False),
            ("intent_classifier", intent_classifier, lambda x: x.is_ready() if x else False),
            ("entity_extractor", entity_extractor, lambda x: x.is_ready() if x else False),
            ("sentiment_analyzer", sentiment_analyzer, lambda x: x.is_ready() if x else False),
            ("response_generator", response_generator, lambda x: x.is_ready() if x else False),
            ("embeddings_manager", embeddings_manager, lambda x: x.is_ready() and x.is_connected() if x else False),
            ("cache_manager", cache_manager, lambda x: x.is_active() if x else False),
        ]

        unhealthy_components = []

        for name, component, check_func in component_checks:
            try:
                is_healthy = check_func(component)
                health_status["components"][name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "available": component is not None
                }

                if not is_healthy:
                    unhealthy_components.append(name)
                    health_status["status"] = "degraded"

            except Exception as e:
                health_status["components"][name] = {
                    "status": "error",
                    "available": component is not None,
                    "error": str(e)
                }
                unhealthy_components.append(name)

        # Add additional details
        health_status["unhealthy_components"] = unhealthy_components
        health_status["healthy_components_count"] = len(component_checks) - len(unhealthy_components)
        health_status["total_components_count"] = len(component_checks)

        return health_status

    except Exception as e:
        logger.error(
            "overall_health_check_error",
            error=str(e),
            exc_info=True
        )
        return {
            "service": "ai-nlp-service",
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }


@router.get("/detailed", summary="Detailed Health Check")
async def detailed_health_check():
    """
    Get detailed health status of all components.
    """
    try:
        detailed_status = {
            "service": "ai-nlp-service",
            "timestamp": time.time(),
            "detailed_components": {},
            "performance_metrics": _get_performance_metrics(),
            "resource_usage": _get_resource_usage()
        }

        # NLP Engine detailed health
        if nlp_engine:
            try:
                detailed_status["detailed_components"]["nlp_engine"] = await nlp_engine.health_check()
            except Exception as e:
                detailed_status["detailed_components"]["nlp_engine"] = {
                    "healthy": False,
                    "error": str(e)
                }

        # Other components detailed health
        component_details = [
            ("text_processor", text_processor),
            ("intent_classifier", intent_classifier),
            ("entity_extractor", entity_extractor),
            ("sentiment_analyzer", sentiment_analyzer),
            ("response_generator", response_generator),
            ("embeddings_manager", embeddings_manager),
            ("cache_manager", cache_manager),
        ]

        for name, component in component_details:
            try:
                if component and hasattr(component, 'health_check'):
                    detailed_status["detailed_components"][name] = await component.health_check()
                else:
                    # Basic status check
                    is_ready = getattr(component, 'is_ready', lambda: False)()
                    is_active = getattr(component, 'is_active', lambda: False)()
                    detailed_status["detailed_components"][name] = {
                        "healthy": is_ready or is_active,
                        "ready": is_ready,
                        "active": is_active,
                        "available": component is not None
                    }

                    # Add component-specific stats
                    if hasattr(component, 'get_cache_stats'):
                        detailed_status["detailed_components"][name]["cache_stats"] = await component.get_cache_stats()
                    elif hasattr(component, 'get_stats'):
                        detailed_status["detailed_components"][name]["stats"] = await component.get_stats()

            except Exception as e:
                detailed_status["detailed_components"][name] = {
                    "healthy": False,
                    "error": str(e)
                }

        return detailed_status

    except Exception as e:
        logger.error(
            "detailed_health_check_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detailed health check failed"
        )


@router.get("/ready", summary="Readiness Check")
async def readiness_check():
    """
    Check if the service is ready to handle requests.
    """
    try:
        # Check critical components
        critical_checks = [
            ("nlp_engine", nlp_engine, lambda x: x.is_ready() if x else False),
            ("cache_manager", cache_manager, lambda x: x.is_active() if x else False),
        ]

        for name, component, check_func in critical_checks:
            try:
                is_ready = check_func(component)
                if not is_ready:
                    return {
                        "ready": False,
                        "reason": f"Critical component {name} is not ready",
                        "timestamp": time.time()
                    }
            except Exception as e:
                return {
                    "ready": False,
                    "reason": f"Critical component {name} check failed: {str(e)}",
                    "timestamp": time.time()
                }

        return {
            "ready": True,
            "timestamp": time.time(),
            "components_checked": [name for name, _, _ in critical_checks]
        }

    except Exception as e:
        logger.error(
            "readiness_check_error",
            error=str(e),
            exc_info=True
        )
        return {
            "ready": False,
            "reason": f"Readiness check failed: {str(e)}",
            "timestamp": time.time()
        }


@router.get("/live", summary="Liveness Check")
async def liveness_check():
    """
    Check if the service is alive (basic liveness probe).
    """
    try:
        # Simple liveness check - service is running
        return {
            "alive": True,
            "timestamp": time.time(),
            "service": "ai-nlp-service"
        }

    except Exception as e:
        logger.error(
            "liveness_check_error",
            error=str(e)
        )
        return {
            "alive": False,
            "timestamp": time.time(),
            "error": str(e)
        }


@router.get("/metrics", summary="Performance Metrics")
async def performance_metrics():
    """
    Get detailed performance metrics.
    """
    try:
        return {
            "timestamp": time.time(),
            "performance": _get_performance_metrics(),
            "resource_usage": _get_resource_usage(),
            "component_metrics": await _get_component_metrics()
        }

    except Exception as e:
        logger.error(
            "performance_metrics_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance metrics"
        )


@router.get("/components/{component_name}", summary="Component Health")
async def component_health(component_name: str):
    """
    Get health status of a specific component.
    """
    try:
        component_map = {
            "nlp_engine": nlp_engine,
            "text_processor": text_processor,
            "intent_classifier": intent_classifier,
            "entity_extractor": entity_extractor,
            "sentiment_analyzer": sentiment_analyzer,
            "response_generator": response_generator,
            "embeddings_manager": embeddings_manager,
            "cache_manager": cache_manager,
        }

        if component_name not in component_map:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Component '{component_name}' not found"
            )

        component = component_map[component_name]

        if not component:
            return {
                "component": component_name,
                "healthy": False,
                "available": False,
                "timestamp": time.time()
            }

        try:
            if hasattr(component, 'health_check'):
                health = await component.health_check()
                return {
                    "component": component_name,
                    **health,
                    "timestamp": time.time()
                }
            else:
                # Basic status check
                is_ready = getattr(component, 'is_ready', lambda: False)()
                is_active = getattr(component, 'is_active', lambda: False)()
                return {
                    "component": component_name,
                    "healthy": is_ready or is_active,
                    "ready": is_ready,
                    "active": is_active,
                    "available": True,
                    "timestamp": time.time()
                }

        except Exception as e:
            return {
                "component": component_name,
                "healthy": False,
                "available": True,
                "error": str(e),
                "timestamp": time.time()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "component_health_error",
            component_name=component_name,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health for component {component_name}"
        )


def _get_service_start_time() -> float:
    """Get service start time (mock implementation)"""
    # In a real implementation, this would track actual start time
    return time.time() - 3600  # Assume started 1 hour ago


def _get_system_info() -> Dict[str, Any]:
    """Get system information"""
    try:
        return {
            "platform": psutil.platform.platform(),
            "python_version": psutil.sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available
        }
    except Exception:
        return {}


def _get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics"""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [],
        }
    except Exception:
        return {}


def _get_resource_usage() -> Dict[str, Any]:
    """Get detailed resource usage"""
    try:
        process = psutil.Process()
        return {
            "process_cpu_percent": process.cpu_percent(),
            "process_memory_info": process.memory_info()._asdict(),
            "process_memory_percent": process.memory_percent(),
            "process_create_time": process.create_time(),
            "process_num_threads": process.num_threads(),
            "process_status": process.status(),
        }
    except Exception:
        return {}


async def _get_component_metrics() -> Dict[str, Any]:
    """Get component-specific metrics"""
    metrics = {}

    # Cache manager metrics
    if cache_manager and hasattr(cache_manager, 'get_cache_stats'):
        try:
            metrics["cache_manager"] = await cache_manager.get_cache_stats()
        except Exception:
            metrics["cache_manager"] = {"error": "Failed to get cache stats"}

    # Embeddings manager metrics
    if embeddings_manager and hasattr(embeddings_manager, 'get_embeddings_stats'):
        try:
            metrics["embeddings_manager"] = await embeddings_manager.get_embeddings_stats()
        except Exception:
            metrics["embeddings_manager"] = {"error": "Failed to get embeddings stats"}

    # NLP engine metrics
    if nlp_engine:
        try:
            metrics["nlp_engine"] = {
                "loaded_models": nlp_engine.get_loaded_models(),
                "is_ready": nlp_engine.is_ready()
            }
        except Exception:
            metrics["nlp_engine"] = {"error": "Failed to get NLP engine stats"}

    return metrics
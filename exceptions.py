"""
NLP Service Exceptions
Custom exception classes for the AI & NLP Service
"""
from fastapi import HTTPException, status
from typing import Any, Dict, Optional


class NLPException(Exception):
    """Base exception class for NLP service"""

    def __init__(
        self,
        message: str,
        error_code: str = "NLP_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException"""
        return HTTPException(
            status_code=self.status_code,
            detail={
                "error_code": self.error_code,
                "message": self.message,
                **self.details
            }
        )


class IntentClassificationError(NLPException):
    """Exception for intent classification errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="INTENT_CLASSIFICATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class EntityExtractionError(NLPException):
    """Exception for entity extraction errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ENTITY_EXTRACTION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class SentimentAnalysisError(NLPException):
    """Exception for sentiment analysis errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SENTIMENT_ANALYSIS_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class LanguageDetectionError(NLPException):
    """Exception for language detection errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="LANGUAGE_DETECTION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class ResponseGenerationError(NLPException):
    """Exception for response generation errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RESPONSE_GENERATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class EmbeddingsGenerationError(NLPException):
    """Exception for embeddings generation errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="EMBEDDINGS_GENERATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class CacheError(NLPException):
    """Exception for cache-related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class ModelNotReadyError(NLPException):
    """Exception for when models are not ready"""

    def __init__(self, component_name: str, details: Optional[Dict[str, Any]] = None):
        message = f"Component '{component_name}' is not ready"
        super().__init__(
            message=message,
            error_code="MODEL_NOT_READY",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={**({"component": component_name}, **(details or {}))}
        )


class InvalidInputError(NLPException):
    """Exception for invalid input parameters"""

    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="INVALID_INPUT",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={**({"field": field} if field else {}, **(details or {}))}
        )


class UnsupportedLanguageError(NLPException):
    """Exception for unsupported language operations"""

    def __init__(self, language: str, supported_languages: list, details: Optional[Dict[str, Any]] = None):
        message = f"Language '{language}' is not supported"
        super().__init__(
            message=message,
            error_code="UNSUPPORTED_LANGUAGE",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={
                "language": language,
                "supported_languages": supported_languages,
                **(details or {})
            }
        )


class ModelLoadError(NLPException):
    """Exception for model loading errors"""

    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        message = f"Failed to load model '{model_name}'"
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={**({"model": model_name}, **(details or {}))}
        )


class TextProcessingError(NLPException):
    """Exception for text processing errors"""

    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TEXT_PROCESSING_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={**({"operation": operation} if operation else {}, **(details or {}))}
        )


class VectorDatabaseError(NLPException):
    """Exception for vector database errors"""

    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VECTOR_DATABASE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={**({"operation": operation} if operation else {}, **(details or {}))}
        )


class RateLimitError(NLPException):
    """Exception for rate limiting"""

    def __init__(self, limit: int, window: int, details: Optional[Dict[str, Any]] = None):
        message = f"Rate limit exceeded: {limit} requests per {window} seconds"
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={
                "limit": limit,
                "window": window,
                **(details or {})
            }
        )


class ConfigurationError(NLPException):
    """Exception for configuration errors"""

    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={**({"config_key": config_key} if config_key else {}, **(details or {}))}
        )


def setup_exception_handlers(app):
    """Setup FastAPI exception handlers"""

    @app.exception_handler(NLPException)
    async def nlp_exception_handler(request, exc: NLPException):
        """Handle custom NLP exceptions"""
        return exc.to_http_exception()

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc: ValueError):
        """Handle ValueError"""
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "INVALID_VALUE",
                "message": str(exc)
            }
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request, exc: KeyError):
        """Handle KeyError"""
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "MISSING_KEY",
                "message": f"Missing required key: {str(exc)}"
            }
        )

    @app.exception_handler(ConnectionError)
    async def connection_error_handler(request, exc: ConnectionError):
        """Handle ConnectionError"""
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "CONNECTION_ERROR",
                "message": "Service temporarily unavailable due to connection issues"
            }
        )

    @app.exception_handler(TimeoutError)
    async def timeout_error_handler(request, exc: TimeoutError):
        """Handle TimeoutError"""
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error_code": "TIMEOUT_ERROR",
                "message": "Request timed out"
            }
        )
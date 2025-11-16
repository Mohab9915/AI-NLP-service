"""
Sentiment Analysis Router
API endpoints for sentiment analysis and emotion detection
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time

from shared.utils.logger import get_service_logger

router = APIRouter()
logger = get_service_logger("sentiment_router")


# Pydantic models
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze sentiment")
    language: str = Field(default="english", description="Language code")
    include_emotions: bool = Field(default=True, description="Include emotion detection")
    include_keywords: bool = Field(default=True, description="Include sentiment keywords")
    use_lexicon: bool = Field(default=True, description="Use lexicon-based analysis")
    granularity: str = Field(default="document", description="Analysis level: document, sentence, aspect")


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    language: str = Field(default="english", description="Language code")
    include_emotions: bool = Field(default=True, description="Include emotion detection")
    include_keywords: bool = Field(default=True, description="Include sentiment keywords")
    use_lexicon: bool = Field(default=True, description="Use lexicon-based analysis")


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    polarity: float
    subjectivity: float
    emotions: Dict[str, float]
    keywords: List[str]
    language_detected: str
    processing_time_ms: float
    textblob_used: bool
    lexicon_used: bool


# Global variables (will be injected by main app)
sentiment_analyzer = None


async def get_sentiment_analyzer():
    """Dependency to get sentiment analyzer instance"""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analyzer not available"
        )
    return sentiment_analyzer


@router.post("/analyze", response_model=SentimentResponse, summary="Analyze Text Sentiment")
async def analyze_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks,
    analyzer=Depends(get_sentiment_analyzer)
):
    """
    Analyze the sentiment of a single text.

    - **text**: The text to analyze sentiment
    - **language**: Target language (english, arabic, hebrew)
    - **include_emotions**: Include emotion detection
    - **include_keywords**: Include sentiment keywords
    - **use_lexicon**: Use lexicon-based analysis
    - **granularity**: Analysis level (document, sentence, aspect)
    """
    start_time = time.time()

    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        # Analyze sentiment
        result = await analyzer.analyze_sentiment(
            text=request.text,
            language=request.language,
            include_emotions=request.include_emotions,
            include_keywords=request.include_keywords,
            use_lexicon=request.use_lexicon,
            granularity=request.granularity
        )

        processing_time = time.time() - start_time

        # Log analysis result
        background_tasks.add_task(
            logger.info,
            "sentiment_analyzed",
            text_length=len(request.text),
            sentiment=result.get("sentiment"),
            confidence=result.get("confidence"),
            language=request.language,
            processing_time=processing_time
        )

        response = SentimentResponse(
            sentiment=result.get("sentiment", "neutral"),
            confidence=result.get("confidence", 0.0),
            polarity=result.get("polarity", 0.0),
            subjectivity=result.get("subjectivity", 0.0),
            emotions=result.get("emotions", {}),
            keywords=result.get("keywords", []),
            language_detected=result.get("language_detected", request.language),
            processing_time_ms=processing_time * 1000,
            textblob_used=result.get("textblob_used", False),
            lexicon_used=result.get("lexicon_used", False)
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "sentiment_analysis_error",
            text_length=len(request.text),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sentiment analysis failed"
        )


@router.post("/analyze-batch", summary="Analyze Multiple Text Sentiments")
async def analyze_batch_sentiments(
    request: BatchSentimentRequest,
    background_tasks: BackgroundTasks,
    analyzer=Depends(get_sentiment_analyzer)
):
    """
    Analyze sentiments for multiple texts in batch.

    - **texts**: List of texts to analyze sentiment
    - **language**: Target language
    - **include_emotions**: Include emotion detection
    - **include_keywords**: Include sentiment keywords
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
        results = await analyzer.batch_analyze_sentiments(
            texts=request.texts,
            language=request.language,
            include_emotions=request.include_emotions,
            include_keywords=request.include_keywords,
            use_lexicon=request.use_lexicon
        )

        processing_time = time.time() - start_time

        # Log batch processing
        background_tasks.add_task(
            logger.info,
            "batch_sentiment_analyzed",
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
            "batch_sentiment_analysis_error",
            texts_count=len(request.texts),
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch sentiment analysis failed"
        )


@router.post("/emotions", summary="Detect Text Emotions")
async def detect_emotions(
    text: str,
    language: str = "english",
    analyzer=Depends(get_sentiment_analyzer)
):
    """
    Detect emotions in text without full sentiment analysis.
    """
    try:
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        emotions = await analyzer.detect_emotions(text, language)

        return {
            "emotions": emotions,
            "dominant_emotion": max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral",
            "emotion_scores": emotions,
            "language": language
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "emotion_detection_error",
            text_length=len(text),
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Emotion detection failed"
        )


@router.post("/compare", summary="Compare Text Sentiments")
async def compare_sentiments(
    text1: str,
    text2: str,
    language: str = "english",
    analyzer=Depends(get_sentiment_analyzer)
):
    """
    Compare the sentiment of two texts.
    """
    try:
        if not text1.strip() or not text2.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both texts cannot be empty"
            )

        comparison = await analyzer.compare_sentiments(text1, text2, language)

        return comparison

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "sentiment_comparison_error",
            text1_length=len(text1),
            text2_length=len(text2),
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sentiment comparison failed"
        )


@router.get("/emotions", summary="Get Supported Emotions")
async def get_supported_emotions(analyzer=Depends(get_sentiment_analyzer)):
    """
    Get list of supported emotions that the analyzer can detect.
    """
    try:
        emotions = analyzer.get_supported_emotions()

        return {
            "supported_emotions": emotions,
            "total_count": len(emotions),
            "supported_languages": analyzer.get_supported_languages()
        }

    except Exception as e:
        logger.error(
            "get_supported_emotions_error",
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get supported emotions"
        )


@router.get("/lexicon", summary="Get Sentiment Lexicon")
async def get_sentiment_lexicon(
    language: str = "english",
    analyzer=Depends(get_sentiment_analyzer)
):
    """
    Get sentiment lexicon used for analysis.
    """
    try:
        lexicon = analyzer.get_sentiment_lexicon(language)

        return {
            "language": language,
            "positive_words": lexicon.get("positive", []),
            "negative_words": lexicon.get("negative", []),
            "neutral_words": lexicon.get("neutral", []),
            "total_words": len(lexicon.get("positive", [])) + len(lexicon.get("negative", [])) + len(lexicon.get("neutral", []))
        }

    except Exception as e:
        logger.error(
            "get_sentiment_lexicon_error",
            language=language,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get sentiment lexicon"
        )


@router.post("/train", summary="Train Sentiment Analyzer")
async def train_sentiment_analyzer(
    training_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    analyzer=Depends(get_sentiment_analyzer)
):
    """
    Train the sentiment analyzer with new data.
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
            _train_analyzer_background,
            analyzer,
            training_data
        )

        processing_time = time.time() - start_time

        return {
            "message": "Training started in background",
            "examples_count": len(examples),
            "processing_time_ms": processing_time * 1000,
            "estimated_duration": "2-4 minutes"
        }

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "train_sentiment_analyzer_error",
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start sentiment analyzer training"
        )


@router.get("/health", summary="Sentiment Analyzer Health")
async def sentiment_analyzer_health(analyzer=Depends(get_sentiment_analyzer)):
    """
    Check the health status of the sentiment analyzer.
    """
    try:
        health_status = await analyzer.health_check()

        return {
            "service": "sentiment_analyzer",
            "healthy": health_status.get("healthy", False),
            "details": health_status,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(
            "sentiment_analyzer_health_error",
            error=str(e),
            exc_info=True
        )
        return {
            "service": "sentiment_analyzer",
            "healthy": False,
            "error": str(e),
            "timestamp": time.time()
        }


async def _train_analyzer_background(analyzer, training_data: Dict[str, Any]):
    """Background task for training the analyzer"""
    try:
        logger.info(
            "sentiment_analyzer_training_started",
            examples_count=len(training_data.get("examples", []))
        )

        # Train the analyzer
        success = await analyzer.train(training_data)

        if success:
            logger.info("sentiment_analyzer_training_completed_successfully")
        else:
            logger.error("sentiment_analyzer_training_failed")

    except Exception as e:
        logger.error(
            "sentiment_analyzer_training_background_error",
            error=str(e),
            exc_info=True
        )
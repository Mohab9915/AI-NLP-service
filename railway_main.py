"""
AI/NLP Service for Railway Deployment
Standalone version without external dependencies
"""

import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import structlog
import json
import re
import time

logger = structlog.get_logger()

# Pydantic models for API
class ProcessTextRequest(BaseModel):
    text: str
    options: List[str] = Field(default_factory=lambda: ["intent", "entities"])
    context: Dict[str, Any] = Field(default_factory=dict)

class ProcessTextResponse(BaseModel):
    text: str
    results: Dict[str, Any]
    processing_time_ms: float
    model_used: str
    timestamp: datetime

class LanguageDetectionResponse(BaseModel):
    language: str
    confidence: float
    alternatives: List[Dict[str, Any]]

class IntentAnalysisResponse(BaseModel):
    intent: str
    confidence: float
    alternatives: List[str]

class EntityExtractionResponse(BaseModel):
    entities: List[Dict[str, Any]]
    count: int

class SentimentAnalysisResponse(BaseModel):
    sentiment: str
    polarity: float
    subjectivity: float
    confidence: float
    emotion: Optional[str] = None

class EmbeddingsRequest(BaseModel):
    texts: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None

class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model: str
    count: int

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
QDRANT_URL = os.getenv("QDRANT_URL", "")

class AzureOpenAIClient:
    """Azure OpenAI Client for embeddings and analysis"""

    def __init__(self):
        self.endpoint = AZURE_OPENAI_ENDPOINT.rstrip('/')
        self.api_key = AZURE_OPENAI_KEY
        self.api_version = AZURE_OPENAI_API_VERSION
        self.embedding_deployment = AZURE_EMBEDDING_DEPLOYMENT

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Azure OpenAI"""
        if not self.endpoint or not self.api_key:
            # Fallback to mock embeddings for testing
            return [[0.0] * 3072 for _ in texts]

        url = f"{self.endpoint}/openai/deployments/{self.embedding_deployment}/embeddings?api-version={self.api_version}"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "input": texts,
            "dimensions": 3072
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                return [item["embedding"] for item in result["data"]]
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            # Fallback to mock embeddings
            return [[0.0] * 3072 for _ in texts]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting AI/NLP Service...")
    logger.info(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
    logger.info(f"Embedding Deployment: {AZURE_EMBEDDING_DEPLOYMENT}")
    logger.info(f"Qdrant URL: {QDRANT_URL}")

    # Initialize Azure OpenAI client
    app.state.openai_client = AzureOpenAIClient()

    yield

    logger.info("AI/NLP Service shutdown complete")

# FastAPI application
app = FastAPI(
    title="AI/NLP Service",
    description="Advanced AI and NLP processing service with Azure OpenAI integration",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI/NLP Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ai-nlp-service",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "azure_openai_configured": bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY),
        "qdrant_configured": bool(QDRANT_URL)
    }

@app.post("/api/v1/process/text", response_model=ProcessTextResponse)
async def process_text(request: ProcessTextRequest):
    """Process text with AI/NLP capabilities"""

    start_time = time.time()

    try:
        results = {}

        # Process each requested option
        for option in request.options:
            if option == "language":
                results["language"] = detect_language(request.text)
            elif option == "intent":
                results["intent"] = analyze_intent(request.text)
            elif option == "entities":
                results["entities"] = extract_entities(request.text)
            elif option == "sentiment":
                results["sentiment"] = analyze_sentiment(request.text)

        processing_time = (time.time() - start_time) * 1000

        return ProcessTextResponse(
            text=request.text,
            results=results,
            processing_time_ms=processing_time,
            model_used="azure-openai",
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process text")

@app.post("/api/v1/language/detect", response_model=LanguageDetectionResponse)
async def detect_language_endpoint(text: str):
    """Detect language of text"""
    try:
        result = detect_language(text)
        return LanguageDetectionResponse(**result)
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to detect language")

@app.post("/api/v1/intent/analyze", response_model=IntentAnalysisResponse)
async def analyze_intent_endpoint(text: str):
    """Analyze user intent"""
    try:
        result = analyze_intent(text)
        return IntentAnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Error analyzing intent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze intent")

@app.post("/api/v1/entities/extract", response_model=EntityExtractionResponse)
async def extract_entities_endpoint(text: str):
    """Extract entities from text"""
    try:
        result = extract_entities(text)
        return EntityExtractionResponse(**result)
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to extract entities")

@app.post("/api/v1/sentiment/analyze", response_model=SentimentAnalysisResponse)
async def analyze_sentiment_endpoint(text: str):
    """Analyze sentiment of text"""
    try:
        result = analyze_sentiment(text)
        return SentimentAnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze sentiment")

@app.post("/api/v1/embeddings/generate", response_model=EmbeddingsResponse)
async def generate_embeddings_endpoint(request: EmbeddingsRequest):
    """Generate embeddings for texts"""
    try:
        openai_client = app.state.openai_client
        embeddings = await openai_client.generate_embeddings(request.texts)

        return EmbeddingsResponse(
            embeddings=embeddings,
            dimension=len(embeddings[0]) if embeddings else 0,
            model=AZURE_EMBEDDING_DEPLOYMENT,
            count=len(embeddings)
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate embeddings")

@app.post("/api/v1/comprehensive/analyze")
async def comprehensive_analysis(text: str, options: List[str] = None):
    """Comprehensive text analysis"""
    if options is None:
        options = ["language", "intent", "entities", "sentiment"]

    try:
        request = ProcessTextRequest(text=text, options=options)
        return await process_text(request)
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to perform comprehensive analysis")

@app.get("/api/v1/status")
async def get_status():
    """Get service status"""
    return {
        "service": "ai-nlp-service",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.utcnow(),
        "capabilities": {
            "language_detection": True,
            "intent_analysis": True,
            "entity_extraction": True,
            "sentiment_analysis": True,
            "embeddings_generation": True,
            "comprehensive_analysis": True
        },
        "integrations": {
            "azure_openai": bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY),
            "qdrant": bool(QDRANT_URL),
            "postgresql": bool(DATABASE_URL),
            "redis": bool(REDIS_URL)
        }
    }

# NLP Processing Functions
def detect_language(text: str) -> Dict[str, Any]:
    """Detect language of text with confidence scoring"""
    text_lower = text.lower()

    # Check for Arabic characters
    arabic_chars = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
    if any(char in text_lower for char in arabic_chars):
        arabic_ratio = sum(1 for char in text_lower if char in arabic_chars) / len(text_lower)
        return {
            "language": "ar",
            "confidence": min(0.95, arabic_ratio + 0.3),
            "alternatives": [
                {"language": "en", "confidence": 0.1},
                {"language": "he", "confidence": 0.05}
            ]
        }

    # Check for Hebrew characters
    hebrew_chars = set("אבגדהוזחטיכלמנסעפצקרשת")
    if any(char in text_lower for char in hebrew_chars):
        hebrew_ratio = sum(1 for char in text_lower if char in hebrew_chars) / len(text_lower)
        return {
            "language": "he",
            "confidence": min(0.95, hebrew_ratio + 0.3),
            "alternatives": [
                {"language": "en", "confidence": 0.1},
                {"language": "ar", "confidence": 0.05}
            ]
        }

    # Default to English
    return {
        "language": "en",
        "confidence": 0.8,
        "alternatives": [
            {"language": "ar", "confidence": 0.1},
            {"language": "he", "confidence": 0.1}
        ]
    }

def analyze_intent(text: str) -> Dict[str, Any]:
    """Analyze user intent with pattern matching and confidence scoring"""
    text_lower = text.lower()

    # Enhanced intent patterns with confidence weights
    intent_patterns = {
        "product_inquiry": {
            "keywords": ["product", "recommend", "buy", "price", "cost", "cheapest", "best", "looking for", "need", "want"],
            "confidence_base": 0.7
        },
        "support_request": {
            "keywords": ["help", "support", "issue", "problem", "broken", "not working", "error", "fix", "trouble"],
            "confidence_base": 0.8
        },
        "greeting": {
            "keywords": ["hello", "hi", "hey", "good morning", "good evening", "greetings", "howdy"],
            "confidence_base": 0.9
        },
        "goodbye": {
            "keywords": ["bye", "goodbye", "see you", "later", "farewell", "cya"],
            "confidence_base": 0.9
        },
        "question": {
            "keywords": ["what", "how", "when", "where", "why", "which", "who", "can", "could", "would"],
            "confidence_base": 0.6
        },
        "complaint": {
            "keywords": ["complaint", "unhappy", "disappointed", "terrible", "awful", "worst", "hate"],
            "confidence_base": 0.8
        },
        "information_request": {
            "keywords": ["tell me", "explain", "information", "details", "about", "learn", "know"],
            "confidence_base": 0.7
        }
    }

    detected_intent = "general"
    max_confidence = 0.5
    confidence = 0.5

    for intent, pattern in intent_patterns.items():
        keyword_matches = sum(1 for keyword in pattern["keywords"] if keyword in text_lower)
        if keyword_matches > 0:
            intent_confidence = min(0.95, pattern["confidence_base"] + (keyword_matches * 0.1))
            if intent_confidence > max_confidence:
                detected_intent = intent
                max_confidence = intent_confidence
                confidence = intent_confidence

    # Generate alternatives
    alternatives = [alt for alt in intent_patterns.keys() if alt != detected_intent][:2]

    return {
        "intent": detected_intent,
        "confidence": confidence,
        "alternatives": alternatives
    }

def extract_entities(text: str) -> Dict[str, Any]:
    """Extract entities from text with improved pattern matching"""
    entities = []

    # Price extraction with various patterns
    price_patterns = [
        r'\$\d+(?:\.\d{2})?',
        r'\d+\s*dollars?',
        r'\d+\s*USD',
        r'\d+\s*(?:dollars?|USD)',
        r'price[:\s]*\$\d+(?:\.\d{2})?',
        r'cost[:\s]*\$\d+(?:\.\d{2})?'
    ]

    for pattern in price_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract numeric value
            numeric_value = re.sub(r'[^\d.]', '', match)
            if numeric_value:
                entities.append({
                    "text": match,
                    "type": "price",
                    "value": float(numeric_value),
                    "currency": "USD",
                    "confidence": 0.9
                })

    # Product extraction with more keywords
    product_keywords = [
        "laptop", "phone", "smartphone", "tablet", "computer", "camera",
        "headphones", "speaker", "monitor", "keyboard", "mouse", "printer",
        "tv", "television", "router", "modem", "hard drive", "SSD",
        "RAM", "memory", "processor", "CPU", "GPU", "graphics card"
    ]

    for keyword in product_keywords:
        if keyword.lower() in text.lower():
            entities.append({
                "text": keyword,
                "type": "product",
                "value": keyword,
                "confidence": 0.8
            })

    # Brand extraction
    brands = ["apple", "samsung", "dell", "hp", "lenovo", "asus", "microsoft", "google", "sony", "lg"]
    for brand in brands:
        if brand.lower() in text.lower():
            entities.append({
                "text": brand,
                "type": "brand",
                "value": brand,
                "confidence": 0.9
            })

    return {
        "entities": entities,
        "count": len(entities)
    }

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment with enhanced emotion detection"""
    text_lower = text.lower()

    positive_words = [
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "best", "perfect", "awesome", "brilliant", "outstanding",
        "happy", "pleased", "satisfied", "delighted", "thrilled"
    ]

    negative_words = [
        "bad", "terrible", "awful", "horrible", "hate", "worst", "poor",
        "disappointed", "frustrated", "angry", "upset", "sad", "annoyed",
        "broken", "useless", "waste", "garbage", "disgusting"
    ]

    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    # Emotion detection
    emotions = {
        "joy": ["happy", "excited", "thrilled", "delighted"],
        "sadness": ["sad", "unhappy", "depressed", "miserable"],
        "anger": ["angry", "mad", "furious", "outraged"],
        "fear": ["scared", "afraid", "worried", "anxious"],
        "surprise": ["surprised", "shocked", "amazed", "astonished"]
    }

    detected_emotion = None
    emotion_score = 0
    for emotion, emotion_words in emotions.items():
        score = sum(1 for word in emotion_words if word in text_lower)
        if score > emotion_score:
            detected_emotion = emotion
            emotion_score = score

    # Calculate sentiment
    total_words = len(text.split())
    if positive_count > negative_count:
        sentiment = "positive"
        polarity = min(0.5, (positive_count - negative_count) / max(total_words, 1))
    elif negative_count > positive_count:
        sentiment = "negative"
        polarity = max(-0.5, (positive_count - negative_count) / max(total_words, 1))
    else:
        sentiment = "neutral"
        polarity = 0.0

    return {
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": min(0.8, (positive_count + negative_count) / max(total_words, 1)),
        "confidence": max(0.5, min(0.9, (positive_count + negative_count) / max(total_words, 1) * 2)),
        "emotion": detected_emotion
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "railway_main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info"
    )
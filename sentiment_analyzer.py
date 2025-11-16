"""
Sentiment Analyzer - Advanced Sentiment Analysis System
Analyzes text sentiment, emotion, and customer satisfaction indicators
"""
import re
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import textblob
import numpy as np

from shared.config.settings import get_settings
from shared.utils.logger import get_service_logger

settings = get_settings("ai-nlp-service")
logger = get_service_logger("sentiment_analyzer")


class SentimentAnalyzer:
    """Advanced sentiment analysis system"""

    def __init__(self, settings):
        self.settings = settings
        self.is_ready = False
        self.vocabulary = self._load_sentiment_vocabulary()
        self.emotion_keywords = self._load_emotion_keywords()
        self.language_config = self._load_language_config()
        self.lexicon = None  # Will be loaded

    async def initialize(self):
        """Initialize sentiment analyzer"""
        try:
            await self._load_sentiment_lexicon()
            await self._initialize_textblob()
            self.is_ready = True
            logger.info("sentiment_analyzer_initialized")

        except Exception as e:
            logger.error(
                "sentiment_analyzer_initialization_failed",
                error=str(e),
                exc_info=True
            )

    async def cleanup(self):
        """Cleanup sentiment analyzer"""
        # Cleanup any resources
        pass

    def is_ready(self) -> bool:
        """Check if analyzer is ready"""
        return self.is_ready

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return list(self.language_config.keys())

    async def analyze_sentiment(
        self,
        text: str,
        language: str = "english",
        include_emotions: bool = True,
        include_keywords: bool = True,
        use_lexicon: bool = True
    ) -> Dict[str, Any]:
        """Analyze sentiment of text"""

        start_time = time.time()

        try:
            results = {
                "text": text,
                "language": language,
                "sentiment": "neutral",
                "confidence": 0.0,
                "polarity": 0.0,
                "subjectivity": 0.0,
                "emotions": [],
                "keywords": [],
                "analysis_method": "unknown",
                "processing_time_ms": 0
            }

            # TextBlob analysis (primary method)
            blob = self._get_textblob(text)
            if blob:
                results["sentiment"] = blob.sentiment.classification
                results["polarity"] = blob.sentiment.polarity
                results["subjectivity"] = blob.sentiment.subjectivity
                results["confidence"] = abs(blob.sentiment.polarity)

            # Emotion analysis
            if include_emotions:
                emotions = self._analyze_emotions(text, language)
                results["emotions"] = emotions

                # Update overall sentiment if strong emotions detected
                if emotions:
                    emotion_sentiment = self._emotions_to_sentiment(emotions)
                    if emotion_sentiment != "neutral":
                        results["sentiment"] = emotion_sentiment
                        # Adjust confidence based on emotion strength
                        emotion_strength = max([e.get("confidence", 0) for e in emotions])
                        results["confidence"] = max(results["confidence"], emotion_strength)

            # Keyword analysis
            if include_keywords:
                keywords = self._analyze_keywords(text, language)
                results["keywords"] = keywords

                # Adjust sentiment based on keywords
                keyword_sentiment, keyword_confidence = self._keywords_to_sentiment(keywords)
                if keyword_sentiment != "neutral" and keyword_confidence > 0.6:
                    results["sentiment"] = keyword_sentiment
                    results["confidence"] = max(results["confidence"], keyword_confidence)

            # Lexicon-based analysis (if available)
            if use_lexicon and self.lexicon:
                lexicon_results = self._analyze_with_lexicon(text, language)
                if lexicon_results["confidence"] > results["confidence"]:
                    results.update(lexicon_results)

            # Determine analysis method
            if self.lexicon and use_lexicon:
                results["analysis_method"] = "lexicon_enhanced"
            elif results["emotions"] or results["keywords"]:
                results["analysis_method"] = "enhanced"
            else:
                results["analysis_method"] = "textblob"

            # Clamp values
            results["polarity"] = max(min(results["polarity"], 1.0), -1.0)
            results["subjectivity"] = max(min(results["subjectivity"], 1.0), 0.0)
            results["confidence"] = max(min(results["confidence"], 1.0), 0.0)

            results["processing_time_ms"] = (time.time() - start_time) * 1000

            logger.info(
                "sentiment_analyzed",
                sentiment=results["sentiment"],
                polarity=results["polarity"],
                confidence=results["confidence"],
                method=results["analysis_method"],
                processing_time=results["processing_time_ms"]
            )

            return results

        except Exception as e:
            logger.error(
                "sentiment_analysis_error",
                error=str(e),
                exc_info=True
            )

            return {
                "text": text,
                "sentiment": "neutral",
                "confidence": 0.0,
                "polarity": 0.0,
                "subjectivity": 0.0,
                "emotions": [],
                "keywords": [],
                "analysis_method": "failed",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }

    def _get_textblob(self, text: str) -> Optional[textblob.TextBlob]:
        """Get TextBlob object for text"""
        try:
            return textblob.TextBlob(text)
        except Exception as e:
            logger.warning(
                "textblob_creation_error",
                error=str(e)
            )
            return None

    def _analyze_emotions(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Analyze emotions in text"""

        emotions = []
        text_lower = text.lower()

        try:
            emotion_keywords = self.emotion_keywords.get(language, {})
            english_keywords = self.emotion_keywords.get("english", {})
            all_keywords = {**english_keywords, **emotion_keywords}

            # Count emotion keywords
            emotion_counts = {}
            for emotion, keywords in all_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                if count > 0:
                    emotion_counts[emotion] = count

            # Calculate emotion scores
            total_keywords = sum(emotion_counts.values())
            if total_keywords > 0:
                for emotion, count in emotion_counts.items():
                    confidence = count / total_keywords
                    intensity = min(count / len(all_keywords.get(emotion, [])), 1.0)

                    emotions.append({
                        "emotion": emotion,
                        "confidence": confidence,
                        "intensity": intensity,
                        "keyword_count": count,
                        "keywords_found": [k for k in all_keywords[emotion] if k in text_lower]
                    })

            # Sort by confidence
            emotions.sort(key=lambda x: x["confidence"], reverse=True)

        except Exception as e:
            logger.warning(
                "emotion_analysis_error",
                error=str(e)
            )

        return emotions

    def _analyze_keywords(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Analyze sentiment keywords in text"""

        keywords = []
        text_lower = text.lower()

        try:
            # Load language-specific keywords
            positive_keywords = self.vocabulary.get(language, {}).get("positive", [])
            negative_keywords = self.vocabulary.get(language, {}).get("negative", [])
            english_positive = self.vocabulary.get("english", {}).get("positive", [])
            english_negative = self.vocabulary.get("english", {}).get("negative", [])

            all_positive = set(positive_keywords + english_positive)
            all_negative = set(negative_keywords + english_negative)

            # Count positive and negative words
            positive_count = sum(1 for word in text_lower.split() if word in all_positive)
            negative_count = sum(1 for word in text_lower.split() if word in all_negative)

            # Find actual keywords
            found_positive = [word for word in text_lower.split() if word in all_positive]
            found_negative = [word for word in text_lower.split() if word in all_negative]

            if positive_count > 0:
                keywords.append({
                    "type": "positive",
                    "confidence": min(positive_count / max(len(text.split()), 1), 1.0),
                    "count": positive_count,
                    "keywords": found_positive
                })

            if negative_count > 0:
                keywords.append({
                    "type": "negative",
                    "confidence": min(negative_count / max(len(text.split()), 1), 1.0),
                    "count": negative_count,
                    "keywords": found_negative
                })

        except Exception as e:
            logger.warning(
                "keyword_analysis_error",
                error=str(e)
            )

        return keywords

    def _analyze_with_lexicon(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze sentiment using lexicon"""

        if not self.lexicon:
            return {"confidence": 0.0}

        try:
            text_lower = text.lower()
            words = text_lower.split()

            positive_score = 0
            negative_score = 0

            for word in words:
                if word in self.lexicon.get("positive", {}):
                    positive_score += 1
                elif word in self.lexicon.get("negative", {}):
                    negative_score += 1

            total_words = len(words)
            if total_words == 0:
                return {"confidence": 0.0}

            # Calculate polarity and confidence
            if positive_score > negative_score:
                polarity = positive_score / total_words
                confidence = (positive_score - negative_score) / total_words
                sentiment = "positive"
            elif negative_score > positive_score:
                polarity = -negative_score / total_words
                confidence = (negative_score - positive_score) / total_words
                sentiment = "negative"
            else:
                polarity = 0.0
                confidence = 0.0
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "polarity": polarity,
                "confidence": min(abs(confidence), 1.0),
                "positive_score": positive_score,
                "negative_score": negative_score,
                "total_words": total_words
            }

        except Exception as e:
            logger.warning(
                "lexicon_analysis_error",
                error=str(e)
            )
            return {"confidence": 0.0}

    def _emotions_to_sentiment(self, emotions: List[Dict[str, Any]]) -> str:
        """Convert emotions to overall sentiment"""

        if not emotions:
            return "neutral"

        # Group emotions by sentiment
        positive_emotions = ["joy", "happy", "excited", "pleased", "satisfied", "grateful", "confident", "optimistic", "content"]
        negative_emotions = ["sad", "angry", "frustrated", "disappointed", "worried", "anxious", "fearful", "disgusted", "hate"]

        positive_score = sum(e.get("confidence", 0) for e in emotions if e.get("emotion") in positive_emotions)
        negative_score = sum(e.get("confidence", 0) for e in emotions if e.get("emotion") in negative_emotions)

        if positive_score > negative_score:
            return "positive"
        elif negative_score > positive_score:
            return "negative"
        else:
            return "neutral"

    def _keywords_to_sentiment(self, keywords: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Convert keywords to overall sentiment"""

        if not keywords:
            return "neutral", 0.0

        positive_confidence = sum(k.get("confidence", 0) for k in keywords if k.get("type") == "positive")
        negative_confidence = sum(k.get("confidence", 0) for k in keywords if k.get("type") == "negative")

        if positive_confidence > negative_confidence:
            return "positive", positive_confidence
        elif negative_confidence > positive_confidence:
            return "negative", negative_confidence
        else:
            return "neutral", 0.0

    def _load_sentiment_vocabulary(self) -> Dict[str, Dict]:
        """Load sentiment vocabulary for different languages"""

        return {
            "english": {
                "positive": [
                    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
                    "love", "like", "awesome", "perfect", "beautiful", "nice",
                    "happy", "joy", "pleased", "satisfied", "content", "glad",
                    "delighted", "thrilled", "excited", "optimistic", "confident",
                    "proud", "grateful", "thankful", "appreciative"
                ],
                "negative": [
                    "bad", "terrible", "awful", "horrible", "worst", "disgusting",
                    "hate", "dislike", "angry", "frustrated", "disappointed",
                    "sad", "upset", "worried", "anxious", "fearful", "scared",
                    "confused", "lost", "broken", "failed", "error", "mistake",
                    "wrong", "poor", "cheap", "expensive", "slow", "late"
                ]
            },
            "arabic": {
                "positive": [
                    "جيد", "ممتاز", "رائع", "ممتاز جد",
                    "جميل", "رائعين", "ممتاز جدا",
                    "ممتازة", "سعيد", "مبارك", "ممتاز",
                    "سعيد جدا", "رائعان جيد",
                    "حسن", "جيد جدا", "ممتاز حقا"
                ],
                "negative": [
                    "سيئ", "سيء", "سيئة للغاية",
                    "محبط", "مبط", "محبط جدا",
                    "غضب", "غاضب جدا", "فشل",
                    "فشل", "مشكلة", "مشكلة كبيرة",
                    "سيء للغاية", "سيئ جدا"
                ]
            },
            "hebrew": {
                "positive": [
                    "מצוין", "מצוין מאוד",
                    "נהדר", "נהדר מאוד",
                    "מצוין מאוד",
                    "טוב", "טוב מאוד",
                    "סבר", "סבר מאוד",
                    "אהבה", "אהבה מאוד",
                    "שמח", "שמח במיוד",
                    "מרוצה", "מרוצה מאוד"
                ],
                "negative": [
                    "רע", "רע מאוד", "רע לגמרי",
                    "גרוע", "גרוע מאוד",
                    "שנוא", "שנוא גרוע",
                    "מאכזב", "מאוכזב מאוד",
                    "כועם", "כועם מאוד",
                    "מאכזב לגמרי", "כועם מאוד",
                    "חסרום", "חסרום מאוד"
                ]
            }
        }

    def _load_emotion_keywords(self) -> Dict[str, Dict]:
        """Load emotion keywords for different languages"""

        return {
            "english": {
                "joy": [
                    "joyful", "joyous", "delighted", "pleased", "happy", "excited",
                    "thrilled", "ecstatic", "cheerful", "euphoric", "gleeful",
                    "jubilant", "elated", "overjoyed", "blissful", "contented"
                ],
                "sad": [
                    "sad", "sadness", "sorrow", "unhappy", "depressed",
                    "depressing", "gloomy", "melancholy", "mournful",
                    "sorrowful", "downcast", "downhearted", "disheartened",
                    "despondent", "blue", "downcast", "low-spirited"
                ],
                "angry": [
                    "angry", "anger", "furious", "enraged", "infuriated",
                    "outraged", "irate", "livid", "incensed", "annoyed",
                    "frustrated", "aggravated", "vexed", "exasperated",
                    "resentful", "indignant", "outraged", "furious"
                ],
                "fear": [
                    "scared", "afraid", "fearful", "frightened", "terrified",
                    "anxious", "worried", "concerned", "apprehensive",
                    "nervous", "uneasy", "tense", "stressed", "panicked"
                ],
                "surprise": [
                    "surprised", "amazed", "astonished", "shocked",
                    "stunned", "bewildered", "startled", "astonished",
                    "flabbergasted", "taken_aback", "disconcerted"
                ],
                "disgust": [
                    "disgusted", "repulsed", "revolted", "sickened",
                    "nauseated", "disgusting", "appalled", "horrified",
                    "revolting", "disgusting", "sick"
                ]
            },
            "arabic": {
                "joy": [
                    "سعيد", "فرحان", "سعيد جدا", "مبتهج",
                    "مسرور", "مبتهج جدا", "مبتاعج",
                    "حبور", "حبور جدا"
                ],
                "sad": [
                    "حزن", "حزين", "مكئيب",
                    "بائس", "بائس", "مكتئب جدا",
                    "أسف", "أسيف جدا", "مأسف على"
                ],
                "angry": [
                    "غاضب", "غاضب جدا", "غضب للغاية",
                    "مغاضب", "غضب جدا", "مغاضب للغاية"
                ],
                "fear": [
                    "خائف", "خائف جدا", "مذعوروب",
                    "قلق", "قلق من الخوف"
                ]
            },
            "hebrew": {
                "joy": [
                    "שמח", "שמח מאוד", "שמחה"
                ],
                "sad": [
                    "עצוב", "עצוב לעצוב",
                    "עצוב מאוד", "עצוב לעצוב"
                ],
                "angry": [
                    "כועם", "כועם מאוד",
                    "זועז", "זועז מאוד"
                ]
            }
        }

    def _load_language_config(self) -> Dict[str, Dict]:
        """Load language-specific configuration"""
        return {
            "english": {
                "supported_models": ["textblob", "vader", "nltk"],
                "default_model": "textblob",
                "confidence_threshold": 0.1
            },
            "arabic": {
                "supported_models": ["textblob"],
                "default_model": "textblob",
                "confidence_threshold": 0.1
            },
            "hebrew": {
                "supported_models": ["textblob"],
                "default_model": "textblob",
                "confidence_threshold": 0.1
            }
        }

    async def _load_sentiment_lexicon(self):
        """Load sentiment lexicon"""
        try:
            # This would load from a file or database
            # For now, use basic vocabulary
            self.lexicon = {
                "positive": {
                    "good": 1.0,
                    "great": 1.0,
                    "excellent": 1.0,
                    "love": 1.0,
                    "happy": 1.0
                },
                "negative": {
                    "bad": -1.0,
                    "terrible": -1.0,
                    "hate": -1.0,
                    "angry": -1.0,
                    "sad": -1.0
                }
            }
            logger.info("sentiment_lexicon_loaded")

        except Exception as e:
            logger.warning(
                "sentiment_lexicon_loading_error",
                error=str(e)
            )

    async def _initialize_textblob(self):
        """Initialize TextBlob for sentiment analysis"""
        try:
            # TextBlob doesn't require explicit initialization
            logger.info("textblob_initialized")

        except Exception as e:
            logger.warning(
                "textblob_initialization_error",
                error=str(e)
            )

    async def batch_analyze_sentiment(
        self,
        texts: List[str],
        language: str = "english",
        options: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts"""

        results = []

        try:
            # Process texts concurrently
            tasks = [
                self.analyze_sentiment(text, language, **options or {})
                for text in texts
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "text": texts[i],
                        "sentiment": "error",
                        "confidence": 0.0,
                        "error": str(result),
                        "processing_time_ms": 0
                    })
                else:
                    results.append(result)

        except Exception as e:
            logger.error(
                "batch_sentiment_analysis_error",
                texts_count=len(texts),
                error=str(e),
                exc_info=True
            )
            raise

        return results

    async def get_sentiment_trends(
        self,
        texts: List[str],
        language: str = "english",
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze sentiment trends over time"""

        results = await self.batch_analyze_sentiment(texts, language)

        # Calculate trends
        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }

        for result in results:
            sentiment = result.get("sentiment", "neutral")
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1

        total = sum(sentiment_counts.values())
        if total > 0:
            trends = {
                "positive_percentage": (sentiment_counts["positive"] / total) * 100,
                "negative_percentage": (sentiment_counts["negative"] / total) * 100,
                "neutral_percentage": (sentiment_counts["neutral"] / total) * 100,
                "total_texts": total,
                "sentiment_counts": sentiment_counts,
                "time_window_hours": time_window_hours,
                "timestamp": time.time()
            }

            return trends

        return {
            "positive_percentage": 0.0,
            "negative_percentage": 0.0,
            "neutral_percentage": 100.0,
            "total_texts": len(texts),
            "sentiment_counts": sentiment_counts,
            "time_window_hours": time_window_hours,
            "timestamp": time.time()
        }
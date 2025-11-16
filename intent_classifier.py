"""
Intent Classifier - Advanced Intent Recognition System
Handles multi-language intent classification with confidence scoring
"""
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime

from shared.config.settings import get_settings, AIProviderConfig
from shared.utils.logger import get_service_logger
from .ai_providers import AIProviderManager

settings = get_settings("ai-nlp-service")
logger = get_service_logger("intent_classifier")


class IntentClassifier:
    """Advanced intent classification system"""

    def __init__(self, settings):
        self.settings = settings
        self.ai_manager = AIProviderManager(settings)
        self.is_ready = False
        self.intent_patterns = self._load_intent_patterns()
        self.intent_mapping = self._load_intent_mapping()
        self.language_config = self._load_language_config()

    async def initialize(self):
        """Initialize intent classifier"""
        try:
            await self.ai_manager.initialize()
            await self._load_intent_models()
            self.is_ready = True
            logger.info("intent_classifier_initialized")

        except Exception as e:
            logger.error(
                "intent_classifier_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def cleanup(self):
        """Cleanup intent classifier"""
        if self.ai_manager:
            await self.ai_manager.cleanup()

    def is_ready(self) -> bool:
        """Check if classifier is ready"""
        return self.is_ready and self.ai_manager.is_ready()

    def get_available_intents(self) -> List[str]:
        """Get list of available intents"""
        return list(self.intent_mapping.keys())

    async def classify_intent(
        self,
        text: str,
        context: Dict[str, Any] = None,
        language: str = "english",
        use_patterns: bool = True,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """Classify intent with multiple approaches"""

        start_time = time.time()

        try:
            # Pre-process text
            processed_text = self._preprocess_text(text, language)

            results = {
                "text": text,
                "processed_text": processed_text,
                "language": language,
                "classification_method": "unknown",
                "intent": "unknown",
                "confidence": 0.0,
                "alternative_intents": [],
                "reasoning": "",
                "processing_time_ms": 0
            }

            # Pattern-based classification (fast, rule-based)
            if use_patterns:
                pattern_result = self._classify_with_patterns(processed_text, language)
                if pattern_result["confidence"] > 0.7:  # High confidence pattern match
                    results.update(pattern_result)
                    results["classification_method"] = "pattern"
                    results["processing_time_ms"] = (time.time() - start_time) * 1000
                    return results

                results["alternative_intents"].append(pattern_result)

            # AI-based classification (slower but more accurate)
            if use_ai and self.ai_manager.is_ready():
                try:
                    ai_result = await self._classify_with_ai(
                        processed_text,
                        context,
                        language
                    )

                    # Combine results
                    final_intent, final_confidence = self._combine_results(
                        results.get("alternative_intents", []),
                        ai_result
                    )

                    results.update(ai_result)
                    results["intent"] = final_intent
                    results["confidence"] = final_confidence
                    results["classification_method"] = "ai_enhanced"

                except Exception as e:
                    logger.warning(
                        "ai_classification_failed",
                        error=str(e),
                        fallback_to_patterns=True
                    )

            # Fallback to pattern-based if AI failed
            if results["intent"] == "unknown" and results["alternative_intents"]:
                best_pattern = max(
                    results["alternative_intents"],
                    key=lambda x: x.get("confidence", 0)
                )
                results.update(best_pattern)
                results["classification_method"] = "pattern_fallback"

            results["processing_time_ms"] = (time.time() - start_time) * 1000

            logger.info(
                "intent_classified",
                intent=results["intent"],
                confidence=results["confidence"],
                method=results["classification_method"],
                processing_time=results["processing_time_ms"]
            )

            return results

        except Exception as e:
            logger.error(
                "intent_classification_error",
                error=str(e),
                exc_info=True
            )

            return {
                "text": text,
                "intent": "unknown",
                "confidence": 0.0,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }

    def _preprocess_text(self, text: str, language: str) -> str:
        """Pre-process text for intent classification"""
        try:
            # Convert to lowercase
            processed = text.lower().strip()

            # Remove extra whitespace
            processed = re.sub(r'\s+', ' ', processed)

            # Language-specific preprocessing
            if language == "arabic":
                # Remove diacritics for Arabic
                processed = self._remove_arabic_diacritics(processed)
            elif language == "hebrew":
                # Handle Hebrew specific preprocessing
                processed = self._preprocess_hebrew(processed)
            elif language == "english":
                # Handle English specific preprocessing
                processed = self._preprocess_english(processed)

            return processed

        except Exception as e:
            logger.warning(
                "text_preprocessing_error",
                error=str(e)
            )
            return text.lower().strip()

    def _classify_with_patterns(self, text: str, language: str) -> Dict[str, Any]:
        """Classify intent using pattern matching"""

        best_match = {
            "intent": "unknown",
            "confidence": 0.0,
            "pattern": None,
            "reasoning": ""
        }

        try:
            language_patterns = self.intent_patterns.get(language, {})
            english_patterns = self.intent_patterns.get("english", {})  # Fallback

            patterns = {**english_patterns, **language_patterns}

            for intent_name, intent_config in patterns.items():
                patterns_list = intent_config.get("patterns", [])
                keywords = intent_config.get("keywords", [])

                # Check for pattern matches
                pattern_match = False
                for pattern in patterns_list:
                    if re.search(pattern, text, re.IGNORECASE):
                        pattern_match = True
                        break

                # Check for keyword matches
                keyword_matches = 0
                for keyword in keywords:
                    if keyword.lower() in text:
                        keyword_matches += 1

                # Calculate confidence
                confidence = 0.0
                if pattern_match:
                    confidence = max(confidence, 0.9)  # High confidence for pattern match
                if keyword_matches > 0:
                    keyword_confidence = min(keyword_matches / len(keywords), 0.8)
                    confidence = max(confidence, keyword_confidence)

                # Update best match
                if confidence > best_match["confidence"]:
                    best_match = {
                        "intent": intent_name,
                        "confidence": confidence,
                        "pattern": patterns_list[0] if pattern_match else None,
                        "keyword_matches": keyword_matches,
                        "reasoning": f"Pattern match: {pattern_match}, Keywords: {keyword_matches}/{len(keywords)}"
                    }

        except Exception as e:
            logger.warning(
                "pattern_classification_error",
                error=str(e)
            )

        return best_match

    async def _classify_with_ai(
        self,
        text: str,
        context: Dict[str, Any] = None,
        language: str = "english"
    ) -> Dict[str, Any]:
        """Classify intent using AI"""

        try:
            # Prepare prompt for AI
            system_prompt = self._get_intent_classification_prompt(language)
            user_prompt = self._prepare_intent_prompt(text, context, language)

            # Get available intents
            available_intents = list(self.intent_mapping.keys())

            # Process with AI
            ai_result = await self.ai_manager.process_message(
                message=user_prompt,
                context={
                    "system_prompt": system_prompt,
                    "available_intents": available_intents,
                    "task_type": "intent_classification",
                    "language": language
                }
            )

            # Parse AI response
            parsed_result = self._parse_ai_intent_response(ai_result, available_intents)

            return parsed_result

        except Exception as e:
            logger.error(
                "ai_intent_classification_error",
                error=str(e),
                exc_info=True
            )
            raise

    def _combine_results(
        self,
        pattern_results: List[Dict[str, Any]],
        ai_result: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Combine pattern and AI classification results"""

        try:
            ai_intent = ai_result.get("intent", "unknown")
            ai_confidence = ai_result.get("confidence", 0.0)

            # Find pattern result for the same intent
            pattern_result = None
            for pr in pattern_results:
                if pr.get("intent") == ai_intent:
                    pattern_result = pr
                    break

            if pattern_result:
                # Combine confidences (weighted average)
                combined_confidence = (
                    ai_confidence * 0.7 +  # Weight AI higher
                    pattern_result.get("confidence", 0.0) * 0.3
                )
                return ai_intent, combined_confidence
            else:
                return ai_intent, ai_confidence

        except Exception as e:
            logger.warning(
                "result_combination_error",
                error=str(e)
            )
            return ai_result.get("intent", "unknown"), ai_result.get("confidence", 0.0)

    def _get_intent_classification_prompt(self, language: str) -> str:
        """Get system prompt for intent classification"""

        base_prompt = """
        You are an intent classification expert. Analyze the given message and determine the user's intent.

        Available intents:
        - greeting: User is saying hello or starting conversation
        - farewell: User is saying goodbye or ending conversation
        - price_inquiry: User is asking about prices or costs
        - product_inquiry: User is asking about products or services
        - order_status: User is asking about order status or tracking
        - complaint: User is expressing dissatisfaction or reporting a problem
        - support_request: User needs help or assistance
        - information_request: User is asking for general information
        - confirmation: User is confirming something or agreeing
        - negation: User is denying or disagreeing
        - appointment: User wants to schedule or reschedule
        - payment: User is asking about payment or billing
        - shipping: User is asking about shipping or delivery
        - return: User wants to return or exchange something
        - technical_issue: User is reporting technical problems
        """

        if language == "arabic":
            return base_prompt + """

            قم بتحليل الرسالة باللغة العربية إذا كانت باللغة العربية.
            استجبت بتنسيق JSON مع الحقول التالية:
            {
                "intent": "intent_name",
                "confidence": 0.95,
                "reasoning": "brief explanation"
            }
            """

        elif language == "hebrew":
            return base_prompt + """

            Analyze the message in Hebrew if it's in Hebrew.
            Respond in JSON format with the following fields:
            {
                "intent": "intent_name",
                "confidence": 0.95,
                "reasoning": "brief explanation"
            }
            """

        return base_prompt + """

        Respond in JSON format with the following fields:
        {
            "intent": "intent_name",
            "confidence": 0.95,
            "reasoning": "brief explanation"
        }
        """

    def _prepare_intent_prompt(
        self,
        text: str,
        context: Dict[str, Any] = None,
        language: str = "english"
    ) -> str:
        """Prepare user prompt for intent classification"""

        prompt_parts = [f"Classify the intent of this message: '{text}'"]

        if context:
            context_info = []
            if context.get("recent_messages"):
                recent_msgs = context["recent_messages"][-3:]  # Last 3 messages
                context_info.append("Recent conversation:")
                for i, msg in enumerate(recent_msgs):
                    context_info.append(f"{i+1}. {msg}")

            if context.get("customer_info"):
                context_info.append(f"Customer info: {context['customer_info']}")

            if context_info:
                prompt_parts.append("\nContext: " + "\n".join(context_info))

        return "\n".join(prompt_parts)

    def _parse_ai_intent_response(
        self,
        ai_result: Dict[str, Any],
        available_intents: List[str]
    ) -> Dict[str, Any]:
        """Parse AI response for intent classification"""

        try:
            ai_response = ai_result.get("response", "")

            # Try to extract JSON from response
            if ai_response.startswith("{") and ai_response.endswith("}"):
                try:
                    parsed = json.loads(ai_response)

                    intent = parsed.get("intent", "unknown")
                    confidence = parsed.get("confidence", 0.0)
                    reasoning = parsed.get("reasoning", "")

                    # Validate intent
                    if intent not in available_intents:
                        # Find closest match
                        intent = self._find_closest_intent(intent, available_intents)
                        confidence *= 0.8  # Reduce confidence for fuzzy match

                    return {
                        "intent": intent,
                        "confidence": min(max(confidence, 0.0), 1.0),  # Clamp between 0 and 1
                        "reasoning": reasoning,
                        "provider": ai_result.get("provider"),
                        "model": ai_result.get("model")
                    }

                except json.JSONDecodeError:
                    # Fallback to text analysis
                    return self._parse_text_intent_response(ai_response, available_intents)

            # Fallback to text analysis
            return self._parse_text_intent_response(ai_response, available_intents)

        except Exception as e:
            logger.warning(
                "ai_response_parsing_error",
                error=str(e)
            )
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def _parse_text_intent_response(
        self,
        response: str,
        available_intents: List[str]
    ) -> Dict[str, Any]:
        """Parse text-based AI response"""

        try:
            response_lower = response.lower()

            # Look for intent names in response
            for intent in available_intents:
                if intent.lower() in response_lower:
                    return {
                        "intent": intent,
                        "confidence": 0.7,  # Moderate confidence for text match
                        "reasoning": f"Intent '{intent}' found in AI response"
                    }

            # Fallback
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "reasoning": "No clear intent identified in AI response"
            }

        except Exception as e:
            logger.warning(
                "text_response_parsing_error",
                error=str(e)
            )
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def _find_closest_intent(self, intent: str, available_intents: List[str]) -> str:
        """Find closest matching intent using string similarity"""

        intent_lower = intent.lower().replace("_", "").replace(" ", "")

        best_match = "unknown"
        best_score = 0

        for available_intent in available_intents:
            available_lower = available_intent.lower().replace("_", "").replace(" ", "")

            # Simple similarity check (can be improved with Levenshtein distance)
            score = 0
            if available_lower in intent_lower:
                score = len(available_lower) / len(intent_lower)
            elif intent_lower in available_lower:
                score = len(intent_lower) / len(available_lower)

            if score > best_score:
                best_score = score
                best_match = available_intent

        return best_match

    def _load_intent_patterns(self) -> Dict[str, Dict]:
        """Load intent patterns and keywords"""
        return {
            "english": {
                "greeting": {
                    "patterns": [
                        r"^(hi|hello|hey|good morning|good afternoon|good evening)",
                        r"^(how are you|howdy|what's up)"
                    ],
                    "keywords": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "how are you", "howdy"]
                },
                "price_inquiry": {
                    "patterns": [
                        r"(how much|what.*price|cost|pricing)",
                        r"(how expensive|cheaper|discount)",
                        r"(price.*range|cost.*range)"
                    ],
                    "keywords": ["price", "cost", "pricing", "expensive", "cheap", "discount", "how much", "cost range"]
                },
                "product_inquiry": {
                    "patterns": [
                        r"(tell me about|looking for|searching for)",
                        r"(do you have|do you sell|available)",
                        r"(product.*information|details.*product)"
                    ],
                    "keywords": ["product", "item", "looking for", "searching", "available", "details", "information"]
                },
                "order_status": {
                    "patterns": [
                        r"(order.*status|track.*order|where.*order)",
                        r"(delivery.*status|shipping.*status)",
                        r"(when.*will.*arrive|order.*shipped)"
                    ],
                    "keywords": ["order", "status", "track", "delivery", "shipping", "arrived", "shipped"]
                },
                "complaint": {
                    "patterns": [
                        r"(not working|broken|doesn't work)",
                        r"(terrible|awful|disappointed|unsatisfied)",
                        r"(problem|issue|wrong|error|mistake)"
                    ],
                    "keywords": ["complaint", "problem", "issue", "wrong", "broken", "doesn't work", "terrible", "awful"]
                },
                "support_request": {
                    "patterns": [
                        r"(help.*me|need.*help|can.*you.*help)",
                        r"(support|assistance|customer.*service)",
                        r"(how.*do.*I|what.*should.*I)"
                    ],
                    "keywords": ["help", "support", "assistance", "customer service", "how to", "need help"]
                }
            },
            "arabic": {
                "greeting": {
                    "patterns": [
                        r"^(مرحبا|أهلا|السلام عليكم|مساء الخير)",
                        r"(كيف حالك|أخبارك|كيفك)"
                    ],
                    "keywords": ["مرحبا", "أهلا", "السلام", "كيف حالك", "أخبارك"]
                },
                "price_inquiry": {
                    "patterns": [
                        r"(كم.*السعر|ما.*السعر|كم.*تكلفة)",
                        r"(السعر.*كم|التكلفة.*كم|كم.*الثمن)",
                        r"(غالي|رخيص|تخفيض)"
                    ],
                    "keywords": ["سعر", "تكلفة", "كم", "غالي", "رخيص", "تخفيض", "ثمن"]
                },
                "product_inquiry": {
                    "patterns": [
                        r"(أبحث.*عن|أريد.*شراء|هل.*عندكم)",
                        r"(منتج.*هذا|معلومات.*المنتج)",
                        r"(متوفر|موجود)"
                    ],
                    "keywords": ["منتج", "معلومات", "أبحث", "شراء", "متوفر", "موجود"]
                },
                "order_status": {
                    "patterns": [
                        r"(حالة.*الطلب|متى.*يصل|أين.*طلبي)",
                        r"(توصيل.*الطلب|رقم.*التتبع)",
                        r"(تم.*الشحن|تم.*التوصيل)"
                    ],
                    "keywords": ["طلب", "حالة", "وصول", "توصيل", "رقم تتبع"]
                }
            },
            "hebrew": {
                "greeting": {
                    "patterns": [
                        r"^(שלום|היי|בוקר|בוקר טוב)",
                        r"(מה שלום|איך אתם|מה קורה)"
                    ],
                    "keywords": ["שלום", "היי", "בוקר", "מה שלום", "מה קורה"]
                },
                "price_inquiry": {
                    "patterns": [
                        r"(כמה עוללת|מה המחיר|מה עולה)",
                        r"(יוקר|זול|הנחה|הנחה בבקשה)",
                        r"(טווח מחיר|טווח עלות)"
                    ],
                    "keywords": ["מחיר", "עלות", "כמה", "יוקר", "זול", "טווח", "הנחה"]
                },
                "product_inquiry": {
                    "patterns": [
                        r"(מחפש* אחר|מחפשת* מוצר|חיפוש* מוצר)",
                        r"(יש לכם|אצל אצלכם|האם יש לכם)",
                        r"(פרטים* מידע|מידע על* המוצר)"
                    ],
                    keywords": ["מוצר", "מידע", "פרטים", "מחפש", "יש לכם", "אצלכם"]
                }
            }
        }

    def _load_intent_mapping(self) -> Dict[str, Dict]:
        """Load intent mapping and metadata"""
        return {
            "greeting": {
                "category": "conversation_start",
                "priority": 1,
                "response_type": "greeting",
                "requires_context": False
            },
            "price_inquiry": {
                "category": "business",
                "priority": 2,
                "response_type": "price_info",
                "requires_context": True
            },
            "product_inquiry": {
                "category": "business",
                "priority": 2,
                "response_type": "product_info",
                "requires_context": True
            },
            "order_status": {
                "category": "support",
                "priority": 3,
                "response_type": "order_info",
                "requires_context": True
            },
            "complaint": {
                "category": "support",
                "priority": 4,
                "response_type": "escalation",
                "requires_context": True,
                "escalation_required": True
            },
            "support_request": {
                "category": "support",
                "priority": 3,
                "response_type": "assistance",
                "requires_context": True
            }
        }

    def _load_language_config(self) -> Dict[str, Dict]:
        """Load language-specific configuration"""
        return {
            "english": {
                "text_direction": "ltr",
                "preprocessors": ["normalize_text", "remove_punctuation"],
                "confidence_threshold": 0.7
            },
            "arabic": {
                "text_direction": "rtl",
                "preprocessors": ["remove_diacritics", "normalize_arabic"],
                "confidence_threshold": 0.6
            },
            "hebrew": {
                "text_direction": "rtl",
                "preprocessors": ["normalize_hebrew", "remove_niqqud"],
                "confidence_threshold": 0.6
            }
        }

    def _remove_arabic_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics"""
        arabic_diacritics = {
            '\u064B': '',  # Fatha
            '\u064C': '',  # Damma
            '\u064D': '',  # Kasra
            '\u064E': '',  # Sukun
            '\u0650': '',  # Shadda
            '\u0651': '',  # Sukuun
            '\u0652': '',  # Fathatan
            '\u0653': '',  # Dammatan
            '\u0654': '',  # Kasratan
            '\u064F': '',  # Wasla
            '\u0622': ''   # Maddah
        }
        return text.translate(str.maketrans(arabic_diacritics))

    def _preprocess_hebrew(self, text: str) -> str:
        """Preprocess Hebrew text"""
        # Remove niqqud (vowel points)
        niqqud = {
            '\u05B0': '', '\u05B1': '', '\u05B2': '', '\u05B3': '',
            '\u05B4': '', '\u05B5': '', '\u05B6': '', '\u05B7': '',
            '\u05B8': '', '\u05B9': '', '\u05BB': '', '\u05BC': '',
            '\u05BD': '', '\u05BE': '', '\u05BF': '', '\u05C0': '',
            '\u05C1': '', '\u05C2': '', '\u05C3': '', '\u05C4': '',
            '\u05C5': '', '\u05C6': '', '\u05C7': ''
        }
        return text.translate(str.maketrans(niqqud))

    def _preprocess_english(self, text: str) -> str:
        """Preprocess English text"""
        # Remove extra punctuation and normalize
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def _load_intent_models(self):
        """Load intent classification models"""
        try:
            # Initialize any ML models if available
            # This could include loading pre-trained models
            logger.info("intent_models_loaded_or_skipped")

        except Exception as e:
            logger.warning(
                "intent_model_loading_skipped",
                error=str(e)
            )
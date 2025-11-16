"""
Response Generator - Advanced Context-Aware Response Generation
Generates personalized responses based on intent, entities, sentiment, and conversation context
"""
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import time
import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re

from shared.config.settings import get_settings
from shared.utils.logger import get_service_logger

settings = get_settings("ai-nlp-service")
logger = get_service_logger("response_generator")


class ResponseType(Enum):
    """Types of responses"""
    GREETING = "greeting"
    FAREWELL = "farewell"
    QUESTION = "question"
    STATEMENT = "statement"
    REQUEST = "request"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    APOLOGY = "apology"
    INFORMATION = "information"


class ResponseStyle(Enum):
    """Response styles"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CASUAL = "casual"
    FORMAL = "formal"
    EMPHATIC = "emphatic"
    CONCISE = "concise"
    DETAILED = "detailed"


@dataclass
class ResponseTemplate:
    """Response template with placeholders"""
    template_id: str
    text: str
    response_type: ResponseType
    style: ResponseStyle
    conditions: Dict[str, Any] = field(default_factory=dict)
    placeholders: List[str] = field(default_factory=list)
    priority: int = 1
    language: str = "english"
    context_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedResponse:
    """Generated response with metadata"""
    text: str
    response_type: ResponseType
    style: ResponseStyle
    confidence: float
    template_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[str] = field(default_factory=list)
    processing_time: float = 0.0


class ResponseTemplateManager:
    """Manages response templates and patterns"""

    def __init__(self):
        self.templates: Dict[str, ResponseTemplate] = {}
        self.pattern_responses: Dict[str, List[str]] = {}
        self.load_default_templates()

    def load_default_templates(self):
        """Load default response templates"""

        # English templates
        english_templates = [
            ResponseTemplate(
                template_id="greeting_general",
                text="Hello! How can I help you today?",
                response_type=ResponseType.GREETING,
                style=ResponseStyle.FRIENDLY,
                language="english"
            ),
            ResponseTemplate(
                template_id="greeting_formal",
                text="Good day! Welcome to our service. How may I assist you?",
                response_type=ResponseType.GREETING,
                style=ResponseStyle.FORMAL,
                language="english"
            ),
            ResponseTemplate(
                template_id="farewell_general",
                text="Thank you for contacting us. Have a great day!",
                response_type=ResponseType.FAREWELL,
                style=ResponseStyle.FRIENDLY,
                language="english"
            ),
            ResponseTemplate(
                template_id="product_inquiry",
                text="I'd be happy to help you with information about {product}. {product_details}",
                response_type=ResponseType.REQUEST,
                style=ResponseStyle.HELPFUL,
                placeholders=["product", "product_details"],
                language="english"
            ),
            ResponseTemplate(
                template_id="price_inquiry",
                text="The price of {product} is {price}. {additional_info}",
                response_type=ResponseType.REQUEST,
                style=ResponseStyle.INFORMATIONAL,
                placeholders=["product", "price", "additional_info"],
                language="english"
            ),
            ResponseTemplate(
                template_id="complaint_acknowledgment",
                text="I'm sorry to hear that you're experiencing issues with {issue}. Let me help you resolve this right away.",
                response_type=ResponseType.COMPLAINT,
                style=ResponseStyle.EMPHATIC,
                placeholders=["issue"],
                language="english"
            ),
            ResponseTemplate(
                template_id="clarification_needed",
                text="I want to make sure I understand correctly. Are you asking about {topic}?",
                response_type=ResponseType.CLARIFICATION,
                style=ResponseStyle.PROFESSIONAL,
                placeholders=["topic"],
                language="english"
            ),
            ResponseTemplate(
                template_id="confirmation",
                text="Yes, that's correct. {confirmation_details}",
                response_type=ResponseType.CONFIRMATION,
                style=ResponseStyle.CONFIDENT,
                placeholders=["confirmation_details"],
                language="english"
            ),
            ResponseTemplate(
                template_id="information_request",
                text="I can help you with {information_type}. {details}",
                response_type=ResponseType.INFORMATION,
                style=ResponseStyle.HELPFUL,
                placeholders=["information_type", "details"],
                language="english"
            )
        ]

        # Arabic templates
        arabic_templates = [
            ResponseTemplate(
                template_id="greeting_arabic",
                text="مرحباً! كيف يمكنني مساعدتك اليوم؟",
                response_type=ResponseType.GREETING,
                style=ResponseStyle.FRIENDLY,
                language="arabic"
            ),
            ResponseTemplate(
                template_id="farewell_arabic",
                text="شكراً لتواصلك معنا. أتمنى لك يوماً سعيداً!",
                response_type=ResponseType.FAREWELL,
                style=ResponseStyle.FRIENDLY,
                language="arabic"
            ),
            ResponseTemplate(
                template_id="product_inquiry_arabic",
                text="يسعدني مساعدتك بمعلومات حول {product}. {product_details}",
                response_type=ResponseType.REQUEST,
                style=ResponseStyle.HELPFUL,
                placeholders=["product", "product_details"],
                language="arabic"
            ),
            ResponseTemplate(
                template_id="price_inquiry_arabic",
                text="سعر {product} هو {price}. {additional_info}",
                response_type=ResponseType.REQUEST,
                style=ResponseStyle.INFORMATIONAL,
                placeholders=["product", "price", "additional_info"],
                language="arabic"
            )
        ]

        # Hebrew templates
        hebrew_templates = [
            ResponseTemplate(
                template_id="greeting_hebrew",
                text="שלום! איך אני יכול לעזור לך היום?",
                response_type=ResponseType.GREETING,
                style=ResponseStyle.FRIENDLY,
                language="hebrew"
            ),
            ResponseTemplate(
                template_id="farewell_hebrew",
                text="תודה שפנית אלינו. יום טוב!",
                response_type=ResponseType.FAREWELL,
                style=ResponseStyle.FRIENDLY,
                language="hebrew"
            ),
            ResponseTemplate(
                template_id="product_inquiry_hebrew",
                text="שמח לעזור לך עם מידע על {product}. {product_details}",
                response_type=ResponseType.REQUEST,
                style=ResponseStyle.HELPFUL,
                placeholders=["product", "product_details"],
                language="hebrew"
            )
        ]

        # Add all templates
        for template in english_templates + arabic_templates + hebrew_templates:
            self.templates[template.template_id] = template

        # Pattern-based responses
        self.pattern_responses = {
            # English patterns
            "hello": [
                "Hello! How can I assist you today?",
                "Hi there! What can I help you with?",
                "Good day! How may I help you?"
            ],
            "thanks": [
                "You're welcome!",
                "My pleasure! Is there anything else I can help with?",
                "Happy to help! Let me know if you need anything else."
            ],
            "how_are_you": [
                "I'm doing well, thank you for asking! I'm here to help you.",
                "I'm functioning perfectly and ready to assist you!"
            ],
            "what_can_you_do": [
                "I can help you with product information, pricing, order status, and answer questions about our services.",
                "I'm here to assist with inquiries, provide information, and help resolve any issues you might have."
            ],

            # Arabic patterns
            "مرحبا": [
                "مرحباً بك! كيف يمكنني مساعدتك؟",
                "أهلاً وسهلاً! ما الذي يمكنني مساعدتك به؟"
            ],
            "شكرا": [
                "عفواً! هل هناك شيء آخر يمكنني مساعدتك به؟",
                "يسعدني ذلك! لا تتردد في طلب أي مساعدة إضافية."
            ],

            # Hebrew patterns
            "שלום": [
                "שלום! איך אני יכול לעזור לך?",
                "היי! במה אני יכול לסייע לך?"
            ]
        }

    def get_template(self, template_id: str) -> Optional[ResponseTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)

    def find_matching_templates(
        self,
        response_type: ResponseType,
        language: str = "english",
        style: Optional[ResponseStyle] = None,
        intent: Optional[str] = None
    ) -> List[ResponseTemplate]:
        """Find templates matching criteria"""
        matching = []

        for template in self.templates.values():
            if (template.response_type == response_type and
                template.language == language):

                if style and template.style != style:
                    continue

                matching.append(template)

        # Sort by priority
        matching.sort(key=lambda x: x.priority, reverse=True)
        return matching

    def get_pattern_response(self, pattern_key: str, language: str = "english") -> List[str]:
        """Get pattern-based responses"""
        responses = self.pattern_responses.get(pattern_key, [])

        # Filter by language if possible
        language_specific = []
        for response in responses:
            if language == "arabic" and any('\u0600' <= c <= '\u06FF' for c in response):
                language_specific.append(response)
            elif language == "hebrew" and any('\u0590' <= c <= '\u05FF' for c in response):
                language_specific.append(response)
            elif language == "english" and not any('\u0600' <= c <= '\u06FF' or '\u0590' <= c <= '\u05FF' for c in response):
                language_specific.append(response)

        return language_specific if language_specific else responses


class ContextManager:
    """Manages conversation context and history"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_contexts: Dict[str, Dict[str, Any]] = {}

    def update_context(
        self,
        conversation_id: str,
        message: str,
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]],
        sentiment: Dict[str, Any]
    ):
        """Update conversation context"""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = {
                "history": [],
                "entities_mentioned": set(),
                "topics_discussed": set(),
                "sentiment_trend": [],
                "last_response_time": None,
                "response_count": 0
            }

        context = self.conversation_contexts[conversation_id]

        # Add to history
        context["history"].append({
            "message": message,
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "timestamp": time.time()
        })

        # Trim history if needed
        if len(context["history"]) > self.max_history:
            context["history"] = context["history"][-self.max_history:]

        # Update entities mentioned
        for entity in entities:
            context["entities_mentioned"].add(entity.get("text", ""))

        # Update topics discussed
        if intent.get("intent"):
            context["topics_discussed"].add(intent["intent"])

        # Update sentiment trend
        context["sentiment_trend"].append(sentiment.get("sentiment", "neutral"))
        if len(context["sentiment_trend"]) > 5:
            context["sentiment_trend"] = context["sentiment_trend"][-5:]

        # Update response count and time
        context["last_response_time"] = time.time()
        context["response_count"] += 1

    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation context"""
        return self.conversation_contexts.get(conversation_id, {
            "history": [],
            "entities_mentioned": set(),
            "topics_discussed": set(),
            "sentiment_trend": [],
            "last_response_time": None,
            "response_count": 0
        })

    def get_recent_entities(self, conversation_id: str, entity_type: str = None) -> List[str]:
        """Get recently mentioned entities"""
        context = self.get_context(conversation_id)
        recent_entities = []

        for item in reversed(context["history"][-3:]):
            for entity in item.get("entities", []):
                if entity_type is None or entity.get("label") == entity_type:
                    recent_entities.append(entity.get("text", ""))

        return recent_entities

    def should_refer_to_history(self, conversation_id: str) -> bool:
        """Check if response should refer to conversation history"""
        context = self.get_context(conversation_id)
        return context["response_count"] > 1


class ResponseGenerator:
    """Advanced response generation system"""

    def __init__(self, settings):
        self.settings = settings
        self.template_manager = ResponseTemplateManager()
        self.context_manager = ContextManager()
        self.is_initialized = False

        # Response generation configuration
        self.config = {
            "max_alternatives": 3,
            "min_confidence": 0.3,
            "context_weight": 0.3,
            "template_weight": 0.7,
            "personalization_enabled": True,
            "multilingual_enabled": True
        }

    async def initialize(self):
        """Initialize the response generator"""
        try:
            logger.info("initializing_response_generator")

            # Load custom templates if available
            await self._load_custom_templates()

            self.is_initialized = True
            logger.info("response_generator_initialized")

        except Exception as e:
            logger.error(
                "response_generator_initialization_failed",
                error=str(e),
                exc_info=True
            )
            raise

    async def cleanup(self):
        """Cleanup response generator resources"""
        try:
            self.context_manager.conversation_contexts.clear()
            logger.info("response_generator_cleaned")

        except Exception as e:
            logger.error(
                "response_generator_cleanup_error",
                error=str(e)
            )

    def is_ready(self) -> bool:
        """Check if response generator is ready"""
        return self.is_initialized

    async def generate_response(
        self,
        input_text: str,
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]],
        sentiment: Dict[str, Any],
        context: Dict[str, Any],
        conversation_id: Optional[str],
        language: str = "english",
        style: Optional[ResponseStyle] = None,
        include_alternatives: bool = True
    ) -> Dict[str, Any]:
        """Generate context-aware response"""

        start_time = time.time()

        try:
            # Update conversation context
            if conversation_id:
                self.context_manager.update_context(
                    conversation_id,
                    input_text,
                    intent,
                    entities,
                    sentiment
                )

            # Determine response type
            response_type = self._determine_response_type(intent, sentiment)

            # Determine response style
            if not style:
                style = self._determine_response_style(sentiment, context)

            # Generate primary response
            primary_response = await self._generate_primary_response(
                input_text,
                intent,
                entities,
                sentiment,
                context,
                conversation_id,
                language,
                response_type,
                style
            )

            # Generate alternatives if requested
            alternatives = []
            if include_alternatives:
                alternatives = await self._generate_alternatives(
                    primary_response,
                    intent,
                    entities,
                    language,
                    response_type,
                    style
                )

            processing_time = time.time() - start_time

            result = {
                "response": primary_response.text,
                "response_type": primary_response.response_type.value,
                "style": primary_response.style.value,
                "confidence": primary_response.confidence,
                "template_id": primary_response.template_id,
                "alternatives": alternatives[:self.config["max_alternatives"] - 1],
                "processing_time_ms": processing_time * 1000,
                "metadata": {
                    "conversation_id": conversation_id,
                    "language": language,
                    "context_used": conversation_id is not None,
                    "entities_referenced": len([e for e in entities if e.get("text") in primary_response.text]),
                    "personalized": primary_response.metadata.get("personalized", False)
                }
            }

            logger.info(
                "response_generated_successfully",
                conversation_id=conversation_id,
                response_type=response_type.value,
                confidence=primary_response.confidence,
                processing_time=processing_time
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "response_generation_error",
                conversation_id=conversation_id,
                error=str(e),
                processing_time=processing_time,
                exc_info=True
            )

            # Fallback response
            fallback_response = self._get_fallback_response(language)

            return {
                "response": fallback_response,
                "response_type": "information",
                "style": "professional",
                "confidence": 0.1,
                "template_id": None,
                "alternatives": [],
                "processing_time_ms": processing_time * 1000,
                "metadata": {
                    "fallback": True,
                    "error": str(e)
                }
            }

    def _determine_response_type(
        self,
        intent: Dict[str, Any],
        sentiment: Dict[str, Any]
    ) -> ResponseType:
        """Determine the type of response needed"""

        intent_name = intent.get("intent", "").lower()
        sentiment_type = sentiment.get("sentiment", "").lower()

        # Greeting patterns
        if intent_name in ["greeting", "hello", "hi"]:
            return ResponseType.GREETING

        # Farewell patterns
        if intent_name in ["farewell", "goodbye", "bye"]:
            return ResponseType.FAREWELL

        # Complaint patterns
        if (sentiment_type in ["negative", "very_negative"] or
            "complaint" in intent_name or
            "problem" in intent_name or
            "issue" in intent_name):
            return ResponseType.COMPLAINT

        # Question patterns
        if (intent_name.endswith("_question") or
            "what" in intent_name or
            "how" in intent_name or
            "when" in intent_name or
            "where" in intent_name):
            return ResponseType.QUESTION

        # Request patterns
        if (intent_name.endswith("_inquiry") or
            "information" in intent_name or
            "help" in intent_name):
            return ResponseType.REQUEST

        # Compliment patterns
        if (sentiment_type in ["positive", "very_positive"] and
            ("good" in intent_name or "nice" in intent_name or "great" in intent_name)):
            return ResponseType.COMPLIMENT

        # Default to statement
        return ResponseType.STATEMENT

    def _determine_response_style(
        self,
        sentiment: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ResponseStyle:
        """Determine appropriate response style"""

        sentiment_type = sentiment.get("sentiment", "neutral")
        context_type = context.get("context_type", "general")

        # Adjust style based on sentiment
        if sentiment_type in ["negative", "very_negative"]:
            return ResponseStyle.EMPHATIC
        elif sentiment_type in ["positive", "very_positive"]:
            return ResponseStyle.FRIENDLY
        elif context_type == "formal":
            return ResponseStyle.FORMAL
        elif context_type == "professional":
            return ResponseStyle.PROFESSIONAL
        else:
            return ResponseStyle.FRIENDLY

    async def _generate_primary_response(
        self,
        input_text: str,
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]],
        sentiment: Dict[str, Any],
        context: Dict[str, Any],
        conversation_id: Optional[str],
        language: str,
        response_type: ResponseType,
        style: ResponseStyle
    ) -> GeneratedResponse:
        """Generate the primary response"""

        # Try template-based generation first
        template_response = await self._generate_template_response(
            intent,
            entities,
            language,
            response_type,
            style,
            conversation_id
        )

        if template_response and template_response.confidence > self.config["min_confidence"]:
            return template_response

        # Try pattern-based generation
        pattern_response = await self._generate_pattern_response(
            input_text,
            intent,
            language
        )

        if pattern_response and pattern_response.confidence > self.config["min_confidence"]:
            return pattern_response

        # Generate contextual response
        contextual_response = await self._generate_contextual_response(
            input_text,
            intent,
            entities,
            sentiment,
            conversation_id,
            language
        )

        return contextual_response

    async def _generate_template_response(
        self,
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]],
        language: str,
        response_type: ResponseType,
        style: ResponseStyle,
        conversation_id: Optional[str]
    ) -> Optional[GeneratedResponse]:
        """Generate response using templates"""

        try:
            # Find matching templates
            templates = self.template_manager.find_matching_templates(
                response_type=response_type,
                language=language,
                style=style,
                intent=intent.get("intent")
            )

            if not templates:
                return None

            # Try each template
            for template in templates:
                # Fill placeholders
                filled_text, confidence = self._fill_template_placeholders(
                    template,
                    intent,
                    entities,
                    conversation_id
                )

                if filled_text:
                    return GeneratedResponse(
                        text=filled_text,
                        response_type=template.response_type,
                        style=template.style,
                        confidence=confidence,
                        template_id=template.template_id,
                        metadata={
                            "template_based": True,
                            "personalized": len(entities) > 0
                        }
                    )

            return None

        except Exception as e:
            logger.error(
                "template_response_generation_error",
                error=str(e)
            )
            return None

    def _fill_template_placeholders(
        self,
        template: ResponseTemplate,
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]],
        conversation_id: Optional[str]
    ) -> Tuple[str, float]:
        """Fill template placeholders with actual values"""

        try:
            text = template.text
            filled_placeholders = 0
            total_placeholders = len(template.placeholders)

            # Create context data
            context_data = {
                "product": self._extract_entity_value(entities, "product"),
                "price": self._extract_entity_value(entities, "price"),
                "quantity": self._extract_entity_value(entities, "quantity"),
                "location": self._extract_entity_value(entities, "location"),
                "time": self._extract_entity_value(entities, "time"),
                "person": self._extract_entity_value(entities, "person"),
                "organization": self._extract_entity_value(entities, "organization"),
            }

            # Add intent-specific data
            intent_name = intent.get("intent", "")
            if "product" in intent_name:
                context_data["product_details"] = self._get_product_details(context_data.get("product"))
            elif "price" in intent_name:
                context_data["additional_info"] = self._get_pricing_info(context_data.get("product"))
            elif "help" in intent_name:
                context_data["details"] = self._get_help_information(intent_name)

            # Fill placeholders
            for placeholder in template.placeholders:
                if placeholder in context_data and context_data[placeholder]:
                    text = text.replace(f"{{{placeholder}}}", str(context_data[placeholder]))
                    filled_placeholders += 1

            # Calculate confidence based on filled placeholders
            confidence = (filled_placeholders / max(total_placeholders, 1)) * self.config["template_weight"]

            # Add context bonus if we have conversation history
            if conversation_id and self.context_manager.should_refer_to_history(conversation_id):
                confidence += self.config["context_weight"]

            return text, min(confidence, 1.0)

        except Exception as e:
            logger.error(
                "template_placeholder_filling_error",
                error=str(e)
            )
            return template.text, 0.3

    def _extract_entity_value(self, entities: List[Dict[str, Any]], entity_type: str) -> Any:
        """Extract value of specific entity type"""
        for entity in entities:
            if entity.get("label") == entity_type:
                return entity.get("text") or entity.get("value")
        return None

    def _get_product_details(self, product: Optional[str]) -> str:
        """Get product details (mock implementation)"""
        if not product:
            return "I can help you with information about our products."
        return f"Here are the details for {product}. This is one of our popular items with excellent customer reviews."

    def _get_pricing_info(self, product: Optional[str]) -> str:
        """Get pricing information (mock implementation)"""
        if not product:
            return "Please let me know which product you're interested in for pricing information."
        return f"We offer competitive pricing and special discounts may be available."

    def _get_help_information(self, intent_name: str) -> str:
        """Get help information based on intent"""
        help_topics = {
            "order_help": "I can help you track orders, check delivery status, and handle returns.",
            "payment_help": "I can assist with payment methods, billing inquiries, and refund requests.",
            "account_help": "I can help with account settings, password recovery, and profile updates.",
            "technical_help": "I can provide technical support and troubleshooting assistance."
        }
        return help_topics.get(intent_name, "I'm here to help with any questions or concerns you may have.")

    async def _generate_pattern_response(
        self,
        input_text: str,
        intent: Dict[str, Any],
        language: str
    ) -> Optional[GeneratedResponse]:
        """Generate response based on patterns"""

        try:
            text_lower = input_text.lower()

            # Check for greeting patterns
            greetings = ["hello", "hi", "hey", "مرحبا", "שלום"]
            if any(greeting in text_lower for greeting in greetings):
                responses = self.template_manager.get_pattern_response("hello", language)
                if responses:
                    return GeneratedResponse(
                        text=random.choice(responses),
                        response_type=ResponseType.GREETING,
                        style=ResponseStyle.FRIENDLY,
                        confidence=0.8,
                        metadata={"pattern_based": True, "pattern": "greeting"}
                    )

            # Check for thanks patterns
            thanks_patterns = ["thank", "thanks", "شكر", "תודה"]
            if any(pattern in text_lower for pattern in thanks_patterns):
                responses = self.template_manager.get_pattern_response("thanks", language)
                if responses:
                    return GeneratedResponse(
                        text=random.choice(responses),
                        response_type=ResponseType.STATEMENT,
                        style=ResponseStyle.FRIENDLY,
                        confidence=0.9,
                        metadata={"pattern_based": True, "pattern": "thanks"}
                    )

            # Check for "how are you" patterns
            how_are_patterns = ["how are you", "كيف حالك", "מה שלומך"]
            if any(pattern in text_lower for pattern in how_are_patterns):
                responses = self.template_manager.get_pattern_response("how_are_you", language)
                if responses:
                    return GeneratedResponse(
                        text=random.choice(responses),
                        response_type=ResponseType.STATEMENT,
                        style=ResponseStyle.FRIENDLY,
                        confidence=0.9,
                        metadata={"pattern_based": True, "pattern": "how_are_you"}
                    )

            return None

        except Exception as e:
            logger.error(
                "pattern_response_generation_error",
                error=str(e)
            )
            return None

    async def _generate_contextual_response(
        self,
        input_text: str,
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]],
        sentiment: Dict[str, Any],
        conversation_id: Optional[str],
        language: str
    ) -> GeneratedResponse:
        """Generate contextual response based on conversation history"""

        try:
            # Get conversation context
            context = self.context_manager.get_context(conversation_id) if conversation_id else {}

            # Base contextual responses
            if context.get("response_count", 0) > 1:
                # Refer to previous conversation
                recent_entities = self.context_manager.get_recent_entities(conversation_id)

                if recent_entities:
                    response = f"Based on our previous conversation about {recent_entities[0]}, "
                    if intent.get("intent"):
                        response += self._generate_intent_specific_response(intent, language)
                    else:
                        response += "how can I help you further?"
                else:
                    response = self._generate_intent_specific_response(intent, language)
            else:
                # First interaction
                response = self._generate_intent_specific_response(intent, language)

            return GeneratedResponse(
                text=response,
                response_type=ResponseType.INFORMATION,
                style=ResponseStyle.PROFESSIONAL,
                confidence=0.6,
                metadata={"contextual": True, "history_used": len(context.get("history", []))}
            )

        except Exception as e:
            logger.error(
                "contextual_response_generation_error",
                error=str(e)
            )
            # Return generic response
            return GeneratedResponse(
                text="I'm here to help you. Could you please provide more details about what you need assistance with?",
                response_type=ResponseType.CLARIFICATION,
                style=ResponseStyle.PROFESSIONAL,
                confidence=0.4,
                metadata={"fallback": True}
            )

    def _generate_intent_specific_response(self, intent: Dict[str, Any], language: str) -> str:
        """Generate response specific to intent"""

        intent_name = intent.get("intent", "")

        # Language-specific responses
        if language == "arabic":
            intent_responses = {
                "product_inquiry": "يمكنني مساعدتك في معلومات المنتجات والأسعار",
                "order_status": "يمكنني التحقق من حالة طلبك",
                "customer_service": "يمكنني توصيلك بخدمة العملاء",
                "technical_support": "يمكنني تقديم الدعم الفني لك",
                "complaint": "أفهم مخاوفك وأريد المساعدة في حلها"
            }
        elif language == "hebrew":
            intent_responses = {
                "product_inquiry": "אני יכול לעזור לך עם מידע על מוצרים ומחירים",
                "order_status": "אני יכול לבדוק את סטטוס ההזמנה שלך",
                "customer_service": "אני יכול לחבר אותך לשירות לקוחות",
                "technical_support": "אני יכול לספק תמיכה טכנית",
                "complaint": "אני מבין את הדאגות שלך ורוצה לעזור לפתור אותן"
            }
        else:  # English
            intent_responses = {
                "product_inquiry": "I can help you with product information and pricing",
                "order_status": "I can check your order status for you",
                "customer_service": "I can connect you with customer service",
                "technical_support": "I can provide technical support assistance",
                "complaint": "I understand your concerns and want to help resolve them"
            }

        return intent_responses.get(intent_name, "I'm here to help you with your request.")

    async def _generate_alternatives(
        self,
        primary_response: GeneratedResponse,
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]],
        language: str,
        response_type: ResponseType,
        style: ResponseStyle
    ) -> List[str]:
        """Generate alternative responses"""

        alternatives = []

        try:
            # Get templates with different styles
            for alt_style in [ResponseStyle.FRIENDLY, ResponseStyle.PROFESSIONAL, ResponseStyle.CASUAL]:
                if alt_style == style:
                    continue  # Skip the same style as primary

                alt_templates = self.template_manager.find_matching_templates(
                    response_type=response_type,
                    language=language,
                    style=alt_style,
                    intent=intent.get("intent")
                )

                if alt_templates:
                    template = alt_templates[0]  # Use the highest priority template
                    filled_text, _ = self._fill_template_placeholders(
                        template,
                        intent,
                        entities,
                        None  # No conversation ID for alternatives
                    )

                    if filled_text and filled_text != primary_response.text:
                        alternatives.append(filled_text)

                        if len(alternatives) >= 2:  # Limit alternatives
                            break

            return alternatives

        except Exception as e:
            logger.error(
                "alternatives_generation_error",
                error=str(e)
            )
            return []

    def _get_fallback_response(self, language: str) -> str:
        """Get fallback response for errors"""

        fallback_responses = {
            "english": "I apologize, but I'm having trouble generating a response right now. Please try again or contact customer support.",
            "arabic": "أعتذر، لكن أواجه صعوبة في توليد رد في الوقت الحالي. يرجى المحاولة مرة أخرى أو التواصل مع خدمة العملاء.",
            "hebrew": "אני מתנצל, אבל אני מתקשה לייצר תגובה כרגע. אנא נסה שוב או צור קשר עם שירות הלקוחות."
        }

        return fallback_responses.get(language, fallback_responses["english"])

    async def _load_custom_templates(self):
        """Load custom response templates (placeholder for future implementation)"""
        try:
            # In a real implementation, this would load from database or files
            logger.info("loading_custom_templates")
            # TODO: Implement custom template loading
        except Exception as e:
            logger.error(
                "custom_templates_loading_error",
                error=str(e)
            )
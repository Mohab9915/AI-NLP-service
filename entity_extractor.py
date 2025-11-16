"""
Entity Extractor - Advanced Entity Recognition System
Extracts entities like products, prices, quantities, dates, locations, etc.
"""
import re
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
import spacy

from shared.config.settings import get_settings
from shared.utils.logger import get_service_logger

settings = get_settings("ai-nlp-service")
logger = get_service_logger("entity_extractor")


@dataclass
class Entity:
    """Entity representation"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }


class EntityExtractor:
    """Advanced entity extraction system"""

    def __init__(self, settings):
        self.settings = settings
        self.is_ready = False
        self.nlp_models = {}
        self.patterns = self._load_entity_patterns()
        self.entity_types = self._load_entity_types()
        self.language_config = self._load_language_config()

    async def initialize(self):
        """Initialize entity extractor"""
        try:
            await self._load_nlp_models()
            self.is_ready = True
            logger.info("entity_extractor_initialized")

        except Exception as e:
            logger.error(
                "entity_extractor_initialization_failed",
                error=str(e),
                exc_info=True
            )
            # Don't raise error - entity extractor can work with patterns only

    async def cleanup(self):
        """Cleanup entity extractor"""
        for model in self.nlp_models.values():
            if hasattr(model, 'close'):
                model.close()

    def is_ready(self) -> bool:
        """Check if extractor is ready"""
        return self.is_ready

    def get_supported_entities(self) -> List[str]:
        """Get list of supported entity types"""
        return list(self.entity_types.keys())

    async def extract_entities(
        self,
        text: str,
        intent: Dict[str, Any] = None,
        language: str = "english",
        use_patterns: bool = True,
        use_nlp: bool = True
    ) -> Dict[str, Any]:
        """Extract entities from text"""

        start_time = time.time()

        try:
            results = {
                "text": text,
                "language": language,
                "entities": [],
                "relations": [],
                "extraction_method": "unknown",
                "processing_time_ms": 0
            }

            # Pattern-based extraction
            pattern_entities = []
            if use_patterns:
                pattern_entities = self._extract_with_patterns(text, language)
                results["entities"].extend([entity.to_dict() for entity in pattern_entities])

            # NLP-based extraction
            nlp_entities = []
            if use_nlp and self.nlp_models.get(language):
                try:
                    nlp_entities = await self._extract_with_nlp(text, language)

                    # Merge with pattern entities, avoiding duplicates
                    merged_entities = self._merge_entities(pattern_entities, nlp_entities)
                    results["entities"] = [entity.to_dict() for entity in merged_entities]

                except Exception as e:
                    logger.warning(
                        "nlp_entity_extraction_failed",
                        error=str(e),
                        fallback_to_patterns=True
                    )

            # Extract relations
            if results["entities"]:
                results["relations"] = self._extract_relations(results["entities"], intent)

            # Determine extraction method
            if use_patterns and use_nlp:
                results["extraction_method"] = "hybrid"
            elif use_nlp:
                results["extraction_method"] = "nlp"
            else:
                results["extraction_method"] = "patterns"

            results["processing_time_ms"] = (time.time() - start_time) * 1000

            logger.info(
                "entities_extracted",
                entity_count=len(results["entities"]),
                method=results["extraction_method"],
                processing_time=results["processing_time_ms"]
            )

            return results

        except Exception as e:
            logger.error(
                "entity_extraction_error",
                error=str(e),
                exc_info=True
            )

            return {
                "text": text,
                "entities": [],
                "relations": [],
                "extraction_method": "failed",
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }

    def _extract_with_patterns(self, text: str, language: str) -> List[Entity]:
        """Extract entities using pattern matching"""

        entities = []

        try:
            patterns = self.patterns.get(language, {})
            english_patterns = self.patterns.get("english", {})
            all_patterns = {**english_patterns, **patterns}

            for entity_type, entity_config in all_patterns.items():
                patterns_list = entity_config.get("patterns", [])
                validation_rules = entity_config.get("validation", {})

                for pattern in patterns_list:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity_text = match.group()
                        start, end = match.span()

                        # Validate entity
                        if self._validate_entity(entity_text, entity_type, validation_rules):
                            confidence = self._calculate_pattern_confidence(
                                entity_text, pattern, validation_rules
                            )

                            entities.append(Entity(
                                text=entity_text,
                                label=entity_type,
                                start=start,
                                end=end,
                                confidence=confidence,
                                metadata={
                                    "extraction_method": "pattern",
                                    "pattern": pattern
                                }
                            ))

        except Exception as e:
            logger.warning(
                "pattern_entity_extraction_error",
                error=str(e)
            )

        return entities

    async def _extract_with_nlp(self, text: str, language: str) -> List[Entity]:
        """Extract entities using NLP models"""

        entities = []

        try:
            nlp_model = self.nlp_models.get(language)
            if not nlp_model:
                return entities

            # Process text with NLP model
            doc = nlp_model(text)

            # Extract entities using spaCy
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_, entity_type)
                if entity_type in self.entity_types:
                    confidence = self._calculate_nlp_confidence(ent, doc)

                    entities.append(Entity(
                        text=ent.text,
                        label=entity_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=confidence,
                        metadata={
                            "extraction_method": "nlp",
                            "spacy_label": ent.label_,
                            "spacy_kb_id": ent.kb_id_
                        }
                    ))

        except Exception as e:
            logger.warning(
                "nlp_entity_extraction_error",
                error=str(e)
            )

        return entities

    def _merge_entities(self, pattern_entities: List[Entity], nlp_entities: List[Entity]) -> List[Entity]:
        """Merge entities from pattern and NLP extraction"""

        merged_entities = []
        used_positions = set()

        # Add pattern entities first
        for entity in pattern_entities:
            used_positions.update(range(entity.start, entity.end))
            merged_entities.append(entity)

        # Add NLP entities that don't overlap with pattern entities
        for entity in nlp_entities:
            # Check for overlap
            overlap = any(
                entity.start <= pos < entity.end for pos in used_positions
            )

            if not overlap:
                used_positions.update(range(entity.start, entity.end))
                merged_entities.append(entity)
            else:
                # Could handle overlapping entities here if needed
                pass

        return merged_entities

    def _extract_relations(self, entities: List[Dict[str, Any]], intent: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Extract relations between entities"""

        relations = []

        try:
            # Define relation patterns based on intent
            relation_rules = self._get_relation_rules(intent)

            for rule in relation_rules:
                source_entities = self._find_entities_by_type(entities, rule["source_type"])
                target_entities = self._find_entities_by_type(entities, rule["target_type"])

                for source in source_entities:
                    for target in target_entities:
                        # Check proximity or other conditions
                        if self._check_relation_conditions(source, target, rule):
                            relations.append({
                                "source": source,
                                "target": target,
                                "relation_type": rule["relation_type"],
                                "confidence": self._calculate_relation_confidence(source, target, rule)
                            })

        except Exception as e:
            logger.warning(
                "relation_extraction_error",
                error=str(e)
            )

        return relations

    def _validate_entity(self, text: str, entity_type: str, rules: Dict[str, Any]) -> bool:
        """Validate entity based on rules"""

        if not rules:
            return True

        # Length validation
        if "min_length" in rules and len(text) < rules["min_length"]:
            return False

        if "max_length" in rules and len(text) > rules["max_length"]:
            return False

        # Format validation
        if "pattern" in rules and not re.match(rules["pattern"], text):
            return False

        # Custom validation
        if "custom_validator" in rules:
            # Would implement custom validation logic here
            pass

        return True

    def _calculate_pattern_confidence(self, text: str, pattern: str, rules: Dict[str, Any]) -> float:
        """Calculate confidence for pattern-based entity extraction"""

        base_confidence = 0.8

        # Adjust confidence based on pattern specificity
        if pattern.startswith("^"):
            base_confidence += 0.1  # Start anchor
        if pattern.endswith("$"):
            base_confidence += 0.1  # End anchor

        # Adjust based on text length (longer entities often more reliable)
        length_factor = min(len(text) / 20.0, 0.2)  # Max 0.2 boost for long entities
        base_confidence += length_factor

        return min(base_confidence, 1.0)

    def _calculate_nlp_confidence(self, entity, doc) -> float:
        """Calculate confidence for NLP-based entity extraction"""

        base_confidence = 0.7  # Base confidence for NLP

        # Adjust based on entity type frequency
        entity_type = entity.label_
        if entity_type in ["PERSON", "ORG", "GPE"]:
            base_confidence += 0.1  # Common entities
        elif entity_type in ["PRODUCT", "EVENT", "WORK_OF_ART"]:
            base_confidence += 0.05  # Less common entities

        # Adjust based on entity length
        length_factor = min(len(entity.text) / 10.0, 0.2)
        base_confidence += length_factor

        # Adjust based on context (would need more sophisticated analysis)
        # This could include surrounding words, sentence position, etc.

        return min(base_confidence, 1.0)

    def _calculate_relation_confidence(self, source: Entity, target: Entity, rule: Dict[str, Any]) -> float:
        """Calculate confidence for extracted relation"""

        base_confidence = 0.6

        # Distance penalty (closer entities more likely to be related)
        distance = abs(target.start - source.end)
        distance_penalty = max(0, 1.0 - (distance / 100.0))  # Penalty for distant entities
        base_confidence += distance_penalty * 0.2

        # Same sentence bonus
        if self._are_in_same_sentence(source.text, target.text, rule.get("text", "")):
            base_confidence += 0.2

        return min(base_confidence, 1.0)

    def _find_entities_by_type(self, entities: List[Dict[str, Any]], entity_type: str) -> List[Dict[str, Any]]:
        """Find entities of specific type"""
        return [entity for entity in entities if entity.get("label") == entity_type]

    def _check_relation_conditions(self, source: Entity, target: Entity, rule: Dict[str, Any]) -> bool:
        """Check if entities meet relation conditions"""

        conditions = rule.get("conditions", {})

        # Distance condition
        if "max_distance" in conditions:
            distance = abs(target.start - source.end)
            if distance > conditions["max_distance"]:
                return False

        # Order condition
        if "order" in conditions:
            if conditions["order"] == "before":
                if source.start >= target.start:
                    return False
            elif conditions["order"] == "after":
                if source.start <= target.start:
                    return False

        return True

    def _are_in_same_sentence(self, source_text: str, target_text: str, full_text: str) -> bool:
        """Check if entities are in the same sentence"""

        # Simple implementation - check if there's a period between them
        try:
            source_pos = full_text.find(source_text)
            target_pos = full_text.find(target_text, source_pos)

            if source_pos == -1 or target_pos == -1:
                return False

            text_between = full_text[source_pos:target_pos + len(target_text)]
            return "." not in text_between

        except Exception:
            return False

    def _map_spacy_label(self, spacy_label: str, language: str) -> str:
        """Map spaCy label to custom entity type"""

        # Language-specific mappings
        mappings = {
            "english": {
                "PERSON": "person",
                "ORG": "organization",
                "GPE": "location",
                "PRODUCT": "product",
                "EVENT": "event",
                "WORK_OF_ART": "media",
                "LAW": "legal",
                "LANGUAGE": "language",
                "DATE": "date",
                "TIME": "time",
                "PERCENT": "percentage",
                "MONEY": "money",
                "QUANTITY": "quantity",
                "ORDINAL": "ordinal",
                "CARDINAL": "number"
            },
            "arabic": {
                "PERSON": "person",
                "ORG": "organization",
                "GPE": "location",
                "PRODUCT": "product",
                "EVENT": "event"
            },
            "hebrew": {
                "PERSON": "person",
                "ORG": "organization",
                "GPE": "location",
                "PRODUCT": "product",
                "EVENT": "event"
            }
        }

        language_mapping = mappings.get(language, mappings["english"])
        return language_mapping.get(spacy_label, spacy_label)

    def _load_entity_patterns(self) -> Dict[str, Dict]:
        """Load entity extraction patterns"""

        return {
            "english": {
                "product": {
                    "patterns": [
                        r"(iphone|samsung|galaxy|pixel|macbook|ipad|airpods)",
                        r"(nike|adidas|puma|reebok|under armour)",
                        r"(shirt|pants|shoes|dress|jacket|hoodie)",
                        r"(size\s+(xs|s|m|l|xl|xxl|xxxl))",
                        r"(color\s+(red|blue|green|black|white|yellow|pink|purple|orange|brown|gray|grey))",
                    ],
                    "validation": {
                        "min_length": 2,
                        "max_length": 50
                    }
                },
                "price": {
                    "patterns": [
                        r"\$[\d,.]+",
                        r"(\d+\.?\d*)\s*(dollars?|USD|bucks?)",
                        r"(price|cost|worth):\s*[\d,.]+",
                        r"[\d,.]+\s*(dollars?|USD|bucks?)",
                    ],
                    "validation": {
                        "pattern": r"^[\d.,]+"
                    }
                },
                "quantity": {
                    "patterns": [
                        r"(\d+)\s+(pieces?|items?|units?)",
                        r"(quantity|qty):\s*(\d+)",
                        r"(pack\s+of\s*(\d+))",
                        r"(\d+)-(pack|box|dozen)",
                    ],
                    "validation": {
                        "min_length": 1
                    }
                },
                "phone": {
                    "patterns": [
                        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                        r"\+?\d{1,3}[-.]?\d{3}[-.]?\d{4}\b",
                        r"\(\d{3}[-.]?\d{3}[-.]?\d{4})",
                        r"\(555)\d{7}",
                    ],
                    "validation": {
                        "pattern": r"^[\d+().-+]{7,}$"
                    }
                },
                "email": {
                    "patterns": [
                        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
                    ]
                },
                "date": {
                    "patterns": [
                        r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\b",
                        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s*\d{4}",
                        r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s*\d{4})",
                    ]
                },
                "time": {
                    "patterns": [
                        r"\b(\d{1,2}):\d{2}\s*(am|pm|a\.m\.|p\.m\.)",
                        r"\b(\d{1,2})\s*(o'clock|o'clock)\s*(am|pm)",
                    ]
                }
            },
            "arabic": {
                "product": {
                    "patterns": [
                        r"(ايفون|سامسونج|جلكسي|بكس)",
                        r"(قميص|مقاس|مقاسات|XL|L|M|S)",
                        r"(أحمر|أخضر|أبيض|أزرق|أسود|بنفسجي)",
                        r"(أحمر|أزرق|بنفسجي|بني|بني|رمادي)",
                    ],
                    "validation": {
                        "min_length": 2,
                        "max_length": 30
                    }
                },
                "price": {
                    "patterns": [
                        r"(\d+)\s*(ريال|درهم)",
                        r"(ريال)\s*(\d+)",
                        r"(درهم)\s*(\d+)",
                        r"(السعر:\s*\d+)",
                    ]
                },
                "phone": {
                    "patterns": [
                        r"\b(05\d|06\d|07\d|09\d)\d{7}\b",
                        r"\b0\d{9}\s*\d{8}\s*\d{7}\b"
                    ]
                },
                "number": {
                    "patterns": [
                        r"\b(\d+)\b",
                        r"(\d{1,3}(?:,\d{3})*)\b"
                    ]
                }
            }
        }

    def _load_entity_types(self) -> Dict[str, Dict]:
        """Load entity type definitions"""

        return {
            "product": {
                "category": "commerce",
                "priority": 1,
                "attributes": ["name", "brand", "category", "size", "color", "price"]
            },
            "person": {
                "category": "personal",
                "priority": 2,
                "attributes": ["name", "title", "role"]
            },
            "organization": {
                "category": "business",
                "priority": 2,
                "attributes": ["name", "type", "industry"]
            },
            "location": {
                "category": "geographic",
                "priority": 2,
                "attributes": ["name", "type", "address"]
            },
            "price": {
                "category": "commerce",
                "priority": 1,
                "attributes": ["amount", "currency", "value"]
            },
            "date": {
                "category": "temporal",
                "priority": 3,
                "attributes": ["date", "format", "parsed_date"]
            },
            "time": {
                "category": "temporal",
                "priority": 3,
                "attributes": ["time", "format", "parsed_time"]
            },
            "quantity": {
                "category": "numerical",
                "priority": 2,
                "attributes": ["amount", "unit"]
            },
            "phone": {
                "category": "contact",
                "priority": 2,
                "attributes": ["number", "type", "country_code"]
            },
            "email": {
                "category": "contact",
                "priority": 2,
                "attributes": ["address", "domain"]
            }
        }

    def _load_language_config(self) -> Dict[str, Dict]:
        """Load language-specific configuration"""
        return {
            "english": {
                "text_direction": "ltr",
                "decimal_separator": ".",
                "thousands_separator": ","
            },
            "arabic": {
                "text_direction": "rtl",
                "decimal_separator": ".",
                "thousands_separator": ","
            },
            "hebrew": {
                "text_direction": "rtl",
                "decimal_separator": ".",
                "thousands_separator": ","
            }
        }

    def _get_relation_rules(self, intent: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get relation rules based on intent"""

        intent_name = intent.get("intent", "unknown") if intent else "unknown"

        rules = {
            "price_inquiry": [
                {
                    "source_type": "product",
                    "target_type": "price",
                    "relation_type": "has_price",
                    "conditions": {
                        "max_distance": 50
                    }
                }
            ],
            "product_inquiry": [
                {
                    "source_type": "quantity",
                    "target_type": "product",
                    "relation_type": "quantity_of",
                    "conditions": {
                        "max_distance": 20
                    }
                },
                {
                    "source_type": "size",
                    "target_type": "product",
                    "relation_type": "size_of",
                    "conditions": {
                        "max_distance": 15
                    }
                }
            ],
            "order_status": [
                {
                    "source_type": "order",
                    "target_type": "status",
                    "relation_type": "has_status",
                    "conditions": {
                        "max_distance": 100
                    }
                }
            ],
            "appointment": [
                {
                    "source_type": "date",
                    "target_type": "time",
                    "relation_type": "at_time",
                    "conditions": {
                        "max_distance": 50
                    }
                },
                {
                    "source_type": "time",
                    "target_type": "date",
                    "relation_type": "at_date",
                    "conditions": {
                        "max_distance": 50
                    }
                }
            ]
        }

        return rules.get(intent_name, [])

    async def _load_nlp_models(self):
        """Load NLP models for entity extraction"""
        try:
            # Load spaCy models for supported languages
            supported_languages = ["english"]  # Add Arabic and Hebrew when models are available

            for lang in supported_languages:
                try:
                    # Try to load spaCy model
                    nlp_model = spacy.load(f"{lang}_core_web_sm")
                    self.nlp_models[lang] = nlp_model
                    logger.info(f"spacy_model_loaded", language=lang)

                except OSError:
                    logger.warning(
                        f"spacy_model_not_available",
                        language=lang
                    )
                    # Continue without NLP for this language

        except Exception as e:
            logger.warning(
                "nlp_model_loading_error",
                error=str(e)
            )
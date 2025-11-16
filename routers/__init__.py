"""
NLP Service Routers Package
Contains all API routers for the AI & NLP Service
"""

from . import intent, entities, sentiment, language, response_gen, embeddings, health

__all__ = ["intent", "entities", "sentiment", "language", "response_gen", "embeddings", "health"]
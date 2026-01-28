"""
SMART LLM Module

Provides embedding and title generation backends for Gemini and Ollama.
"""

from smart.llm.embeddings import EmbeddingBackend, GeminiEmbedder, OllamaEmbedder
from smart.llm.titles import TitleGenerator, GeminiTitleGenerator, OllamaTitleGenerator

__all__ = [
    "EmbeddingBackend",
    "GeminiEmbedder",
    "OllamaEmbedder",
    "TitleGenerator",
    "GeminiTitleGenerator", 
    "OllamaTitleGenerator",
]

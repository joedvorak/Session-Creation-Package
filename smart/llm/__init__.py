"""
SMART LLM Module

Provides embedding and title generation backends for Gemini and Ollama.
"""

from smart.llm.embeddings import (
    EmbeddingBackend, 
    GeminiEmbedder, 
    OllamaEmbedder,
    get_ollama_models as get_ollama_embedding_models,
    check_ollama_connection,
)
from smart.llm.titles import (
    TitleGenerator, 
    GeminiTitleGenerator, 
    OllamaTitleGenerator,
    get_ollama_models as get_ollama_generation_models,
)

__all__ = [
    "EmbeddingBackend",
    "GeminiEmbedder",
    "OllamaEmbedder",
    "TitleGenerator",
    "GeminiTitleGenerator", 
    "OllamaTitleGenerator",
    "get_ollama_embedding_models",
    "get_ollama_generation_models",
    "check_ollama_connection",
]

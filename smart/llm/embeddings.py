"""
Embedding backends for SMART system.

Provides abstraction over different embedding providers (Gemini, Ollama)
with consistent interface and caching integration.
"""

import os
import time
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from smart.core.database import EmbeddingCache, EmbeddingConfig


class EmbeddingBackend(ABC):
    """
    Abstract base class for embedding backends.
    
    Implementations should handle:
    - Single and batch embedding generation
    - Rate limiting and retries
    - Model version detection
    """
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass
    
    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the model version string."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array of embedding vector
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def get_config(self, task_type: str = "SEMANTIC_SIMILARITY") -> EmbeddingConfig:
        """Get embedding configuration for this backend."""
        return EmbeddingConfig(
            model_name=self.model_name,
            model_version=self.model_version,
            task_type=task_type
        )


class CachedEmbedder:
    """
    Wrapper that adds caching to any embedding backend.
    
    Usage:
        cache = EmbeddingCache("cache.db")
        backend = GeminiEmbedder(api_key="...")
        embedder = CachedEmbedder(backend, cache)
        
        # Will use cache when available
        embedding = embedder.embed(text)
    """
    
    def __init__(
        self,
        backend: EmbeddingBackend,
        cache: EmbeddingCache,
        task_type: str = "SEMANTIC_SIMILARITY"
    ):
        self.backend = backend
        self.cache = cache
        self.task_type = task_type
        self._config = backend.get_config(task_type)
    
    @property
    def config(self) -> EmbeddingConfig:
        return self._config
    
    def embed(self, text: str) -> np.ndarray:
        """Get embedding, using cache if available."""
        # Check cache first
        cached = self.cache.get_embedding(text, self._config)
        if cached is not None:
            return cached
        
        # Generate new embedding
        embedding = self.backend.embed(text)
        
        # Store in cache
        self.cache.store_embedding(text, embedding, self._config)
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts, using cache where available.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to print progress
            
        Returns:
            List of embedding vectors in same order as input
        """
        # Check cache for all texts
        cached = self.cache.get_embeddings_batch(texts, self._config)
        
        # Identify which need embedding
        to_embed = [t for t in texts if cached[t] is None]
        
        if show_progress:
            print(f"Found {len(texts) - len(to_embed)}/{len(texts)} embeddings in cache")
        
        if to_embed:
            if show_progress:
                print(f"Generating {len(to_embed)} new embeddings...")
            
            # Generate new embeddings
            new_embeddings = self.backend.embed_batch(to_embed)
            
            # Store in cache
            self.cache.store_embeddings_batch(
                list(zip(to_embed, new_embeddings)),
                self._config
            )
            
            # Update cached dict
            for text, emb in zip(to_embed, new_embeddings):
                cached[text] = emb
        
        # Return in original order
        return [cached[t] for t in texts]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats from backend and cache."""
        return {
            "backend": {
                "model_name": self.backend.model_name,
                "model_version": self.backend.model_version,
                "task_type": self.task_type,
            },
            "cache": self.cache.get_stats()
        }


class GeminiEmbedder(EmbeddingBackend):
    """
    Embedding backend using Google Gemini API.
    
    Usage:
        embedder = GeminiEmbedder(api_key="your-api-key")
        embedding = embedder.embed("Hello world")
    """
    
    DEFAULT_MODEL = "gemini-embedding-001"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        task_type: str = "SEMANTIC_SIMILARITY",
        delay_seconds: float = 0.0
    ):
        """
        Initialize Gemini embedder.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model: Model name (default: gemini-embedding-001)
            task_type: Embedding task type for API
            delay_seconds: Delay between API calls for rate limiting
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Provide api_key parameter or set GEMINI_API_KEY environment variable."
            )
        
        self._model = model
        self._task_type = task_type
        self._delay_seconds = delay_seconds
        self._client = None
        self._model_version = None
    
    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
                
                # Try to get model info for version
                try:
                    model_info = self._client.models.get(model=self._model)
                    self._model_version = getattr(model_info, 'version', self._model)
                except Exception:
                    self._model_version = self._model
                    
            except ImportError:
                raise ImportError(
                    "google-genai package required for Gemini embeddings. "
                    "Install with: pip install google-genai"
                )
        return self._client
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def model_version(self) -> str:
        if self._model_version is None:
            self._get_client()  # This sets _model_version
        return self._model_version or self._model
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        client = self._get_client()
        
        response = client.models.embed_content(
            model=self._model,
            contents=[text],
            config={"task_type": self._task_type}
        )
        
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)
        
        return np.array(response.embeddings[0].values, dtype=np.float32)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        client = self._get_client()
        embeddings = []
        
        for text in texts:
            response = client.models.embed_content(
                model=self._model,
                contents=[text],
                config={"task_type": self._task_type}
            )
            embeddings.append(
                np.array(response.embeddings[0].values, dtype=np.float32)
            )
            
            if self._delay_seconds > 0:
                time.sleep(self._delay_seconds)
        
        return embeddings


class OllamaEmbedder(EmbeddingBackend):
    """
    Embedding backend using local Ollama server.
    
    Usage:
        embedder = OllamaEmbedder(model="nomic-embed-text")
        embedding = embedder.embed("Hello world")
    """
    
    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_HOST = "http://localhost:11434"
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0
    ):
        """
        Initialize Ollama embedder.
        
        Args:
            model: Ollama model name
            host: Ollama server URL
            timeout: Request timeout in seconds
        """
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._model_version = None
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Fetch model information from Ollama."""
        import requests
        
        try:
            response = requests.post(
                f"{self._host}/api/show",
                json={"name": self._model},
                timeout=self._timeout
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def model_version(self) -> str:
        if self._model_version is None:
            info = self._get_model_info()
            # Use model digest as version if available
            self._model_version = info.get("digest", self._model)[:16]
        return self._model_version
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        import requests
        
        response = requests.post(
            f"{self._host}/api/embeddings",
            json={
                "model": self._model,
                "prompt": text
            },
            timeout=self._timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama embedding failed: {response.status_code} - {response.text}"
            )
        
        data = response.json()
        return np.array(data["embedding"], dtype=np.float32)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class SentenceTransformerEmbedder(EmbeddingBackend):
    """
    Embedding backend using sentence-transformers library.
    
    Useful for running models locally without Ollama.
    
    Usage:
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        embedding = embedder.embed("Hello world")
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None):
        """
        Initialize sentence-transformers embedder.
        
        Args:
            model: Model name from HuggingFace
            device: Device to run on ('cpu', 'cuda', etc.)
        """
        self._model_name = model
        self._device = device
        self._model = None
        self._model_version = None
    
    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name, device=self._device)
                
                # Try to get version info
                try:
                    import sentence_transformers
                    self._model_version = f"{self._model_name}-st{sentence_transformers.__version__}"
                except Exception:
                    self._model_version = self._model_name
                    
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def model_version(self) -> str:
        if self._model_version is None:
            self._get_model()
        return self._model_version or self._model_name
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        model = self._get_model()
        return model.encode(text, convert_to_numpy=True).astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [e.astype(np.float32) for e in embeddings]


def create_embedder(
    backend: str = "gemini",
    model: Optional[str] = None,
    cache: Optional[EmbeddingCache] = None,
    task_type: str = "SEMANTIC_SIMILARITY",
    **kwargs
) -> EmbeddingBackend | CachedEmbedder:
    """
    Factory function to create an embedder.
    
    Args:
        backend: Backend type ('gemini', 'ollama', 'sentence-transformers')
        model: Model name (uses backend default if not specified)
        cache: Optional EmbeddingCache for caching
        task_type: Task type for embedding configuration
        **kwargs: Additional backend-specific arguments
        
    Returns:
        Embedder instance (CachedEmbedder if cache provided)
    """
    if backend == "gemini":
        embedder = GeminiEmbedder(
            model=model or GeminiEmbedder.DEFAULT_MODEL,
            task_type=task_type,
            **kwargs
        )
    elif backend == "ollama":
        embedder = OllamaEmbedder(
            model=model or OllamaEmbedder.DEFAULT_MODEL,
            **kwargs
        )
    elif backend in ("sentence-transformers", "st", "sbert"):
        embedder = SentenceTransformerEmbedder(
            model=model or SentenceTransformerEmbedder.DEFAULT_MODEL,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'gemini', 'ollama', or 'sentence-transformers'")
    
    if cache is not None:
        return CachedEmbedder(embedder, cache, task_type=task_type)
    
    return embedder

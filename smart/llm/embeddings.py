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


def get_ollama_models(
    host: str = "http://localhost:11434",
    timeout: float = 10.0,
    filter_embedding: bool = False
) -> List[str]:
    """
    Discover available models from Ollama server.
    
    Args:
        host: Ollama server URL
        timeout: Request timeout in seconds
        filter_embedding: If True, only return models known to support embeddings
        
    Returns:
        List of model names, or empty list if Ollama unavailable
    """
    import requests
    
    try:
        response = requests.get(
            f"{host.rstrip('/')}/api/tags",
            timeout=timeout
        )
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            
            if filter_embedding:
                # Known embedding models - filter to only these if present
                # This list can be expanded as new embedding models become available
                embedding_keywords = [
                    "embed", "nomic", "mxbai", "bge", "e5", "gte", "instructor"
                ]
                filtered = [
                    m for m in models 
                    if any(kw in m.lower() for kw in embedding_keywords)
                ]
                # Return filtered if any matches, otherwise return all (user may have renamed)
                return filtered if filtered else models
            
            return models
    except Exception:
        pass
    return []


def check_ollama_connection(host: str = "http://localhost:11434", timeout: float = 5.0) -> bool:
    """
    Check if Ollama server is reachable.
    
    Args:
        host: Ollama server URL
        timeout: Request timeout in seconds
        
    Returns:
        True if Ollama is reachable, False otherwise
    """
    import requests
    
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


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
    
    @property
    def truncation_count(self) -> int:
        """Get truncation count from backend if supported."""
        if hasattr(self.backend, 'truncation_count'):
            return self.backend.truncation_count
        return 0
    
    @property
    def truncated_ids(self) -> List[str]:
        """Get truncated IDs from backend if supported."""
        if hasattr(self.backend, 'truncated_ids'):
            return self.backend.truncated_ids
        return []
    
    def reset_truncation_stats(self):
        """Reset truncation stats on backend if supported."""
        if hasattr(self.backend, 'reset_truncation_stats'):
            self.backend.reset_truncation_stats()
    
    def embed(self, text: str, text_id: Optional[str] = None) -> np.ndarray:
        """Get embedding, using cache if available."""
        # Check cache first
        cached = self.cache.get_embedding(text, self._config)
        if cached is not None:
            return cached
        
        # Generate new embedding (pass text_id if backend supports it)
        if hasattr(self.backend, 'embed') and 'text_id' in self.backend.embed.__code__.co_varnames:
            embedding = self.backend.embed(text, text_id=text_id)
        else:
            embedding = self.backend.embed(text)
        
        # Store in cache
        self.cache.store_embedding(text, embedding, self._config)
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        text_ids: Optional[List[str]] = None,
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts, using cache where available.
        
        Args:
            texts: List of texts to embed
            text_ids: Optional list of identifiers for tracking truncation
            show_progress: Whether to print progress
            
        Returns:
            List of embedding vectors in same order as input
        """
        # Check cache for all texts
        cached = self.cache.get_embeddings_batch(texts, self._config)
        
        # Identify which need embedding (keep track of indices for text_ids)
        to_embed = []
        to_embed_ids = []
        for i, t in enumerate(texts):
            if cached[t] is None:
                to_embed.append(t)
                if text_ids:
                    to_embed_ids.append(text_ids[i])
                else:
                    to_embed_ids.append(None)
        
        if show_progress:
            print(f"Found {len(texts) - len(to_embed)}/{len(texts)} embeddings in cache")
        
        if to_embed:
            if show_progress:
                print(f"Generating {len(to_embed)} new embeddings...")
            
            # Generate new embeddings (pass text_ids if backend supports it)
            if hasattr(self.backend, 'embed_batch'):
                try:
                    new_embeddings = self.backend.embed_batch(to_embed, text_ids=to_embed_ids)
                except TypeError:
                    # Backend doesn't support text_ids parameter
                    new_embeddings = self.backend.embed_batch(to_embed)
            else:
                new_embeddings = [self.backend.embed(t) for t in to_embed]
            
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
        stats = {
            "backend": {
                "model_name": self.backend.model_name,
                "model_version": self.backend.model_version,
                "task_type": self.task_type,
            },
            "cache": self.cache.get_stats()
        }
        
        # Add truncation stats if available
        if hasattr(self.backend, 'truncation_count'):
            stats["truncation_count"] = self.backend.truncation_count
        
        return stats


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
    
    Handles context length limits by automatically truncating text if needed.
    Check `truncation_count` after embedding to see how many texts were truncated.
    """
    
    DEFAULT_MODEL = "nomic-embed-text-v2-moe"
    DEFAULT_HOST = "http://localhost:11434"
    # Default context length estimate (characters) - conservative for most models
    # Most embedding models support 512-8192 tokens; ~4 chars per token average
    DEFAULT_MAX_CHARS = 8000
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
        max_chars: Optional[int] = None
    ):
        """
        Initialize Ollama embedder.
        
        Args:
            model: Ollama model name
            host: Ollama server URL
            timeout: Request timeout in seconds
            max_chars: Maximum characters per text (auto-detected if None)
        """
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._model_version = None
        self._max_chars = max_chars  # Will be auto-adjusted on context errors
        self._truncation_count = 0
        self._truncated_ids: List[str] = []  # Track which texts were truncated
    
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
    
    @property
    def truncation_count(self) -> int:
        """Number of texts that were truncated due to context length limits."""
        return self._truncation_count
    
    @property
    def truncated_ids(self) -> List[str]:
        """List of text identifiers that were truncated."""
        return self._truncated_ids.copy()
    
    def reset_truncation_stats(self):
        """Reset truncation tracking for a new batch."""
        self._truncation_count = 0
        self._truncated_ids = []
    
    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text to max_chars, trying to break at word boundary."""
        if len(text) <= max_chars:
            return text
        
        # Try to break at last space before limit
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # Only use space if it's not too far back
            truncated = truncated[:last_space]
        
        return truncated
    
    def embed(self, text: str, text_id: Optional[str] = None) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
            text_id: Optional identifier for tracking truncation (e.g., abstract_id)
            
        Returns:
            Embedding vector as numpy array
        """
        import requests
        
        # Pre-truncate if we have a known limit
        original_len = len(text)
        if self._max_chars is not None and len(text) > self._max_chars:
            text = self._truncate_text(text, self._max_chars)
            self._truncation_count += 1
            if text_id:
                self._truncated_ids.append(text_id)
        
        max_retries = 3
        current_text = text
        
        for attempt in range(max_retries):
            response = requests.post(
                f"{self._host}/api/embeddings",
                json={
                    "model": self._model,
                    "prompt": current_text
                },
                timeout=self._timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return np.array(data["embedding"], dtype=np.float32)
            
            # Check if it's a context length error
            if response.status_code == 500:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", "")
                    if "context length" in error_msg.lower() or "input length" in error_msg.lower():
                        # Reduce text length and retry
                        new_max = int(len(current_text) * 0.7)  # Reduce by 30%
                        if new_max < 100:
                            raise RuntimeError(
                                f"Text too short to truncate further. Original length: {original_len}"
                            )
                        
                        current_text = self._truncate_text(text, new_max)
                        
                        # Update our learned max_chars for future texts
                        if self._max_chars is None or new_max < self._max_chars:
                            self._max_chars = new_max
                        
                        # Track truncation if not already counted
                        if original_len == len(text):  # Wasn't pre-truncated
                            self._truncation_count += 1
                            if text_id and text_id not in self._truncated_ids:
                                self._truncated_ids.append(text_id)
                        
                        continue  # Retry with shorter text
                except (ValueError, KeyError):
                    pass
            
            # Non-recoverable error
            raise RuntimeError(
                f"Ollama embedding failed: {response.status_code} - {response.text}"
            )
        
        raise RuntimeError(
            f"Failed to embed text after {max_retries} truncation attempts"
        )
    
    def embed_batch(self, texts: List[str], text_ids: Optional[List[Optional[str]]] = None) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            text_ids: Optional list of identifiers for tracking truncation
            
        Returns:
            List of embedding vectors
        """
        ids: List[Optional[str]] = text_ids if text_ids is not None else [None] * len(texts)
        return [self.embed(text, text_id) for text, text_id in zip(texts, ids)]


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

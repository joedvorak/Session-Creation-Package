"""
Shared pytest fixtures for SMART test suite.

Provides:
- In-memory database fixtures
- Mock embedding fixtures (deterministic vectors)
- Sample presentation/session data
- Temporary file management
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart.core.database import EmbeddingCache, ConferenceDB, EmbeddingConfig


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def embedding_cache(tmp_path):
    """Create an in-memory embedding cache for testing."""
    db_path = tmp_path / "test_embeddings.db"
    cache = EmbeddingCache(str(db_path))
    yield cache
    # No close() needed - SQLite connections are managed per-operation


@pytest.fixture
def conference_db(tmp_path):
    """Create an in-memory conference database for testing."""
    db_path = tmp_path / "test_conference.db"
    db = ConferenceDB(str(db_path))
    yield db
    # No close() needed - SQLite connections are managed per-operation


@pytest.fixture
def embedding_config():
    """Standard embedding configuration for tests."""
    return EmbeddingConfig(
        model_name="test-model",
        model_version="v1.0",
        dimensions=384,
    )


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_presentations() -> List[Dict[str, Any]]:
    """Generate sample presentation data for testing."""
    return [
        {
            "abstract_id": f"ABS-{i:03d}",
            "title": f"Test Presentation {i}",
            "abstract": f"This is the abstract text for presentation {i}. " * 10,
            "authors": f"Author {i}",
            "keywords": f"keyword{i}, testing",
        }
        for i in range(1, 51)  # 50 presentations
    ]


@pytest.fixture
def sample_abstract_ids(sample_presentations) -> List[str]:
    """Extract abstract IDs from sample presentations."""
    return [p["abstract_id"] for p in sample_presentations]


@pytest.fixture
def sample_texts(sample_presentations) -> List[str]:
    """Extract combined text for embedding from sample presentations."""
    return [
        f"{p['title']}. {p['abstract']}"
        for p in sample_presentations
    ]


# =============================================================================
# Mock Embedding Fixtures
# =============================================================================

@pytest.fixture
def mock_embeddings(sample_abstract_ids) -> np.ndarray:
    """
    Generate deterministic mock embeddings for testing.
    
    Uses seeded random to ensure reproducibility.
    Creates embeddings with some structure (not purely random)
    to allow meaningful similarity tests.
    """
    np.random.seed(42)
    n_items = len(sample_abstract_ids)
    dimensions = 384
    
    # Create 5 "topic clusters" with some variation
    n_clusters = 5
    cluster_centers = np.random.randn(n_clusters, dimensions)
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    
    embeddings = []
    for i in range(n_items):
        # Assign to cluster based on index
        cluster_idx = i % n_clusters
        # Add noise around cluster center
        noise = np.random.randn(dimensions) * 0.3
        embedding = cluster_centers[cluster_idx] + noise
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    return np.array(embeddings)


@pytest.fixture
def mock_embeddings_small() -> np.ndarray:
    """Small set of embeddings for quick tests."""
    np.random.seed(42)
    n_items = 10
    dimensions = 384
    
    embeddings = np.random.randn(n_items, dimensions)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


# =============================================================================
# Mock Embedder Fixture
# =============================================================================

class MockEmbedder:
    """
    Deterministic mock embedder for testing.
    
    Returns consistent embeddings for the same text input.
    """
    
    def __init__(self, dimensions: int = 384, seed: int = 42):
        self.dimensions = dimensions
        self.seed = seed
        self._cache: Dict[str, np.ndarray] = {}
    
    @property
    def model_name(self) -> str:
        return "mock-embedder"
    
    @property
    def model_version(self) -> str:
        return "v1.0-test"
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate deterministic embeddings based on text hash."""
        embeddings = []
        for text in texts:
            if text not in self._cache:
                # Use text hash as seed for reproducibility
                text_seed = hash(text) % (2**32)
                rng = np.random.RandomState(text_seed)
                embedding = rng.randn(self.dimensions)
                embedding = embedding / np.linalg.norm(embedding)
                self._cache[text] = embedding
            embeddings.append(self._cache[text])
        return np.array(embeddings)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]


@pytest.fixture
def mock_embedder():
    """Provide a mock embedder instance."""
    return MockEmbedder()


# =============================================================================
# Session/Placement Fixtures
# =============================================================================

@pytest.fixture
def sample_sessions() -> List[Dict[str, Any]]:
    """Sample session data for testing."""
    return [
        {
            "session_id": "SESSION-001",
            "title": "Test Session 1",
            "presentation_ids": ["ABS-001", "ABS-002", "ABS-003", "ABS-004", "ABS-005"],
            "is_hybrid": False,
        },
        {
            "session_id": "SESSION-002", 
            "title": "Test Session 2",
            "presentation_ids": ["ABS-006", "ABS-007", "ABS-008", "ABS-009", "ABS-010"],
            "is_hybrid": False,
        },
        {
            "session_id": "HYBRID-001",
            "title": "Hybrid Session",
            "presentation_ids": ["ABS-011", "ABS-012"],
            "is_hybrid": True,
        },
    ]


# =============================================================================
# Temporary File Fixtures
# =============================================================================

@pytest.fixture
def temp_csv_file(tmp_path, sample_presentations):
    """Create a temporary CSV file with sample presentation data."""
    import pandas as pd
    
    df = pd.DataFrame(sample_presentations)
    csv_path = tmp_path / "test_submissions.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_excel_file(tmp_path, sample_presentations):
    """Create a temporary Excel file with sample presentation data."""
    import pandas as pd
    
    df = pd.DataFrame(sample_presentations)
    xlsx_path = tmp_path / "test_submissions.xlsx"
    df.to_excel(xlsx_path, index=False)
    return xlsx_path


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root) -> Path:
    """Return the data directory."""
    return project_root / "data"


# =============================================================================
# Markers for conditional test skipping
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may use files/databases)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (embedding generation, full workflows)"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: Tests requiring Ollama server"
    )
    config.addinivalue_line(
        "markers", "requires_gemini: Tests requiring Gemini API key"
    )

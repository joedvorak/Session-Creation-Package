# SMART Testing Strategy

## Overview

This document outlines the testing strategy for the SMART (Session Matching And Automated Recommendation Tool) project. The goal is to establish automated testing to reduce reliance on manual UI testing and ensure reliability as features are added or modified.

---

## Testing Levels

### 1. Unit Tests
Test individual functions and methods in isolation.

**Focus Areas**:
- Database operations (EmbeddingCache, ConferenceDB)
- Embedding backend methods
- Metrics calculations
- Data transformation functions

### 2. Integration Tests
Test interactions between components.

**Focus Areas**:
- Embedding → Cache → Retrieval flow
- Import → Database → Export flow
- Clustering with actual embeddings
- Title generation with database storage

### 3. End-to-End Tests (Manual for now)
Full workflow validation.

**Focus Areas**:
- Streamlit app wizard flow
- Viewer app functionality
- Export bundle compatibility

---

## Test Infrastructure

### Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_database.py         # EmbeddingCache and ConferenceDB tests
├── test_placement.py        # Clustering algorithm tests (synthetic data)
├── test_placement_real.py   # Regression tests (real AIM26 data)
├── test_embeddings.py       # Embedding backend tests
├── test_metrics.py          # Coherence/distinctiveness tests
├── test_loaders.py          # Import functionality tests
├── test_exporters.py        # Export functionality tests
└── fixtures/
    └── aim26_benchmark.npz  # Optional: pre-extracted for speed (gitignored)

examples/                    # Deidentified AIM26 data (committed to repo)
├── databases/               # Embedding cache + conference DB
├── submissions/             # Raw submission spreadsheet
├── hybrid/                  # Hybrid session pre-assignments
└── exports/                 # Full organizer export
```

**Real-data tests** (`test_placement_real.py`) automatically load from the
example databases shipped in `examples/databases/`. No manual setup is needed
on a fresh clone. For faster repeated runs, generate the `.npz` fixture:
```bash
python scripts/extract_benchmark_fixture.py
```

### Dependencies

Add to `requirements.txt` or `requirements-dev.txt`:

```
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
```

### Configuration

Create `pyproject.toml` or `pytest.ini`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --cov=smart --cov-report=html --cov-report=term-missing"
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

---

## Key Fixtures

### conftest.py

```python
"""Shared test fixtures for SMART tests."""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from smart.core.database import EmbeddingCache, ConferenceDB, EmbeddingConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def embedding_cache(temp_dir):
    """Create an in-memory embedding cache for testing."""
    cache_path = temp_dir / "test_cache.db"
    return EmbeddingCache(cache_path)


@pytest.fixture
def conference_db(temp_dir):
    """Create a test conference database."""
    db_path = temp_dir / "test_working.db"
    return ConferenceDB(db_path, conference_year=26)


@pytest.fixture
def sample_embedding_config():
    """Standard embedding config for tests."""
    return EmbeddingConfig(
        model_name="test-model",
        model_version="v1.0",
        task_type="SEMANTIC_SIMILARITY",
        dimensions=384
    )


@pytest.fixture
def sample_embeddings():
    """Generate deterministic sample embeddings."""
    np.random.seed(42)
    return {
        "text_a": np.random.randn(384).astype(np.float32),
        "text_b": np.random.randn(384).astype(np.float32),
        "text_c": np.random.randn(384).astype(np.float32),
    }


@pytest.fixture
def sample_presentations():
    """Sample presentation data for testing."""
    return [
        {
            "abstract_id": "2600001",
            "title": "Machine Learning for Crop Yield Prediction",
            "abstract": "This study applies ML techniques to predict crop yields.",
            "combined_text": "Machine Learning for Crop Yield Prediction: This study applies ML techniques to predict crop yields.",
        },
        {
            "abstract_id": "2600002",
            "title": "Deep Learning in Weed Detection",
            "abstract": "Neural networks for automated weed identification.",
            "combined_text": "Deep Learning in Weed Detection: Neural networks for automated weed identification.",
        },
        {
            "abstract_id": "2600003",
            "title": "Sensor Networks for Irrigation",
            "abstract": "IoT sensors for precision irrigation management.",
            "combined_text": "Sensor Networks for Irrigation: IoT sensors for precision irrigation management.",
        },
    ]
```

### Mock Embedder

```python
"""Mock embedding backend for deterministic testing."""

import numpy as np
from typing import List
from smart.llm.embeddings import EmbeddingBackend


class MockEmbedder(EmbeddingBackend):
    """
    Deterministic embedder for testing.
    
    Generates consistent embeddings based on text hash.
    Similar texts (by word overlap) produce similar embeddings.
    """
    
    def __init__(self, dimensions: int = 384):
        self._dimensions = dimensions
        self._model_name = "mock-embedder"
        self._model_version = "test-v1"
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def model_version(self) -> str:
        return self._model_version
    
    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic embedding based on text hash."""
        # Use text hash as random seed for reproducibility
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        return np.random.randn(self._dimensions).astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch."""
        return [self.embed(text) for text in texts]


class SimilarityControlledEmbedder(EmbeddingBackend):
    """
    Embedder that allows controlling similarity between texts.
    
    Useful for testing clustering behavior.
    """
    
    def __init__(self, similarity_groups: dict, dimensions: int = 384):
        """
        Args:
            similarity_groups: Dict mapping text to group ID. 
                              Texts in same group will be similar.
        """
        self._groups = similarity_groups
        self._dimensions = dimensions
        self._group_vectors = {}
        self._model_name = "similarity-controlled"
        self._model_version = "test-v1"
        
        # Pre-generate group base vectors
        np.random.seed(42)
        for group_id in set(similarity_groups.values()):
            self._group_vectors[group_id] = np.random.randn(dimensions).astype(np.float32)
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def model_version(self) -> str:
        return self._model_version
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding based on group with small noise."""
        group_id = self._groups.get(text, 0)
        base_vector = self._group_vectors.get(group_id, np.zeros(self._dimensions))
        
        # Add small noise for variation within group
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        noise = np.random.randn(self._dimensions).astype(np.float32) * 0.1
        
        return base_vector + noise
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(text) for text in texts]
```

---

## Test Categories

### 1. Database Tests (test_database.py)

```python
"""Tests for smart.core.database module."""

import pytest
import numpy as np


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""
    
    def test_store_and_retrieve_embedding(self, embedding_cache, sample_embedding_config):
        """Test basic store and retrieve."""
        text = "Test text for embedding"
        embedding = np.random.randn(384).astype(np.float32)
        
        # Store
        embedding_cache.store_embedding(text, embedding, sample_embedding_config)
        
        # Retrieve
        retrieved = embedding_cache.get_embedding(text, sample_embedding_config)
        
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, embedding)
    
    def test_cache_miss_returns_none(self, embedding_cache, sample_embedding_config):
        """Test that missing embedding returns None."""
        result = embedding_cache.get_embedding("nonexistent text", sample_embedding_config)
        assert result is None
    
    def test_different_config_different_cache(self, embedding_cache):
        """Test that different configs have separate cache entries."""
        text = "Same text"
        embedding_v1 = np.random.randn(384).astype(np.float32)
        embedding_v2 = np.random.randn(384).astype(np.float32)
        
        config_v1 = EmbeddingConfig("model", "v1", "SEMANTIC_SIMILARITY")
        config_v2 = EmbeddingConfig("model", "v2", "SEMANTIC_SIMILARITY")
        
        embedding_cache.store_embedding(text, embedding_v1, config_v1)
        embedding_cache.store_embedding(text, embedding_v2, config_v2)
        
        retrieved_v1 = embedding_cache.get_embedding(text, config_v1)
        retrieved_v2 = embedding_cache.get_embedding(text, config_v2)
        
        np.testing.assert_array_almost_equal(retrieved_v1, embedding_v1)
        np.testing.assert_array_almost_equal(retrieved_v2, embedding_v2)
    
    def test_batch_store_and_retrieve(self, embedding_cache, sample_embedding_config):
        """Test batch operations."""
        texts = ["text1", "text2", "text3"]
        embeddings = [np.random.randn(384).astype(np.float32) for _ in texts]
        
        # Store batch
        embedding_cache.store_embeddings_batch(
            list(zip(texts, embeddings)), 
            sample_embedding_config
        )
        
        # Retrieve batch
        results = embedding_cache.get_embeddings_batch(texts, sample_embedding_config)
        
        for text, expected in zip(texts, embeddings):
            np.testing.assert_array_almost_equal(results[text], expected)
    
    def test_get_stats(self, embedding_cache, sample_embedding_config):
        """Test cache statistics."""
        # Store some embeddings
        for i in range(5):
            embedding_cache.store_embedding(
                f"text_{i}",
                np.random.randn(384).astype(np.float32),
                sample_embedding_config
            )
        
        stats = embedding_cache.get_stats()
        
        assert stats["total_embeddings"] == 5
        assert len(stats["models"]) == 1
        assert stats["models"][0]["model_name"] == "test-model"


class TestConferenceDB:
    """Tests for ConferenceDB class."""
    
    def test_import_presentations(self, conference_db, sample_presentations):
        """Test importing presentations."""
        ids = conference_db.import_presentations_batch(sample_presentations)
        
        assert len(ids) == len(sample_presentations)
        
        # Verify retrieval
        presentations = conference_db.get_presentations()
        assert len(presentations) == len(sample_presentations)
    
    def test_create_session(self, conference_db, sample_presentations):
        """Test creating a session."""
        # Import presentations first
        conference_db.import_presentations_batch(sample_presentations)
        pres_ids = [p["abstract_id"] for p in sample_presentations]
        
        # Create session
        conference_db.create_session(
            session_id="S26001",
            title="Test Session",
            presentation_ids=pres_ids[:2]
        )
        
        sessions = conference_db.get_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "S26001"
    
    def test_move_presentation(self, conference_db, sample_presentations):
        """Test moving presentation between sessions."""
        # Setup
        conference_db.import_presentations_batch(sample_presentations)
        pres_ids = [p["abstract_id"] for p in sample_presentations]
        
        conference_db.create_session("S1", "Session 1", pres_ids[:2])
        conference_db.create_session("S2", "Session 2", pres_ids[2:])
        
        # Move first presentation to second session
        conference_db.move_presentation(pres_ids[0], "S2")
        
        # Verify
        session1 = conference_db.get_session("S1")
        session2 = conference_db.get_session("S2")
        
        assert len(session1["presentations"]) == 1
        assert len(session2["presentations"]) == 2
    
    def test_delete_presentation(self, conference_db, sample_presentations):
        """Test deleting a presentation."""
        conference_db.import_presentations_batch(sample_presentations)
        
        initial_count = len(conference_db.get_presentations())
        conference_db.delete_presentation(sample_presentations[0]["abstract_id"])
        
        assert len(conference_db.get_presentations()) == initial_count - 1
```

### 2. Metrics Tests (test_metrics.py)

```python
"""Tests for smart.core.metrics module."""

import pytest
import numpy as np
from smart.core.metrics import calculate_all_metrics


class TestMetrics:
    """Tests for metrics calculations."""
    
    def test_coherence_identical_embeddings(self):
        """Perfect coherence when all embeddings identical."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        abstract_ids = ["1", "2", "3"]
        session_assignments = {"1": "S1", "2": "S1", "3": "S1"}
        
        metrics = calculate_all_metrics(embeddings, abstract_ids, session_assignments)
        
        assert metrics["session_metrics"]["S1"]["coherence"] == pytest.approx(1.0, rel=1e-5)
    
    def test_coherence_orthogonal_embeddings(self):
        """Zero coherence when embeddings orthogonal."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        abstract_ids = ["1", "2", "3"]
        session_assignments = {"1": "S1", "2": "S1", "3": "S1"}
        
        metrics = calculate_all_metrics(embeddings, abstract_ids, session_assignments)
        
        assert metrics["session_metrics"]["S1"]["coherence"] == pytest.approx(0.0, rel=1e-5)
    
    def test_distinctiveness_separate_clusters(self):
        """High distinctiveness when sessions have different topics."""
        embeddings = np.array([
            [1.0, 0.0],  # Session 1
            [1.0, 0.1],  # Session 1
            [0.0, 1.0],  # Session 2
            [0.1, 1.0],  # Session 2
        ])
        abstract_ids = ["1", "2", "3", "4"]
        session_assignments = {"1": "S1", "2": "S1", "3": "S2", "4": "S2"}
        
        metrics = calculate_all_metrics(embeddings, abstract_ids, session_assignments)
        
        # Both sessions should have high distinctiveness
        assert metrics["session_metrics"]["S1"]["distinctiveness"] > 0.8
        assert metrics["session_metrics"]["S2"]["distinctiveness"] > 0.8
```

### 3. Placement Tests (test_placement.py)

```python
"""Tests for smart.core.placement module."""

import pytest
import numpy as np
from smart.core.placement import create_placement_strategy, SessionConstraints


class TestPlacement:
    """Tests for session placement algorithms."""
    
    def test_minimum_session_size_respected(self):
        """All sessions should have at least min_session_size presentations."""
        # Create embeddings with clear clusters
        np.random.seed(42)
        
        # 3 clusters of 10 each
        cluster1 = np.random.randn(10, 64) + np.array([5, 0] + [0]*62)
        cluster2 = np.random.randn(10, 64) + np.array([0, 5] + [0]*62)
        cluster3 = np.random.randn(10, 64) + np.array([-5, -5] + [0]*62)
        
        embeddings = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
        abstract_ids = [f"P{i}" for i in range(30)]
        
        constraints = SessionConstraints(
            min_session_size=8,
            max_session_size=12,
        )
        
        strategy = create_placement_strategy("hybrid_first")
        result = strategy.place(
            embeddings=embeddings,
            abstract_ids=abstract_ids,
            constraints=constraints,
            hybrid_assignments={}
        )
        
        for session in result.sessions:
            assert len(session["presentation_ids"]) >= 8
    
    def test_hybrid_sessions_preserved(self):
        """Hybrid session assignments should be maintained."""
        embeddings = np.random.randn(20, 64).astype(np.float32)
        abstract_ids = [f"P{i}" for i in range(20)]
        
        # Pre-assign some to hybrid session
        hybrid_assignments = {
            "P0": "HYBRID1",
            "P1": "HYBRID1",
            "P2": "HYBRID1",
        }
        
        constraints = SessionConstraints(min_session_size=3, max_session_size=10)
        strategy = create_placement_strategy("hybrid_first")
        
        result = strategy.place(
            embeddings=embeddings,
            abstract_ids=abstract_ids,
            constraints=constraints,
            hybrid_assignments=hybrid_assignments
        )
        
        # Verify hybrid presentations stayed together
        for pres_id, session_id in hybrid_assignments.items():
            assert result.session_assignments[pres_id] == session_id
```

---

## Running Tests

### Basic Run
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_database.py

# Run specific test class
pytest tests/test_database.py::TestEmbeddingCache

# Run specific test
pytest tests/test_database.py::TestEmbeddingCache::test_store_and_retrieve_embedding
```

### With Coverage
```bash
# Generate coverage report
pytest --cov=smart --cov-report=html

# Open HTML report
open htmlcov/index.html
```

### Continuous Integration

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-mock
    
    - name: Run tests
      run: pytest --cov=smart --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

---

## Test Data Management

### Sample Data Files

Create minimal test fixtures that cover edge cases:

1. **sample_presentations.csv**: 10-20 presentations with varied content
2. **sample_hybrid.csv**: Hybrid session data with session column
3. **sample_committees.csv**: 3-5 committees with descriptions

### Synthetic Embedding Generation

For tests requiring specific similarity patterns:

```python
def create_clustered_embeddings(n_clusters, items_per_cluster, dimensions=64):
    """Create embeddings with known cluster structure."""
    embeddings = []
    labels = []
    
    np.random.seed(42)
    
    for cluster_id in range(n_clusters):
        # Random cluster center
        center = np.random.randn(dimensions) * 3
        
        # Items with small variance around center
        for i in range(items_per_cluster):
            embedding = center + np.random.randn(dimensions) * 0.3
            embeddings.append(embedding)
            labels.append(cluster_id)
    
    return np.array(embeddings).astype(np.float32), labels
```

---

## Priority Test Cases

### Must Have Before Release (P0)

1. **Cache config hash consistency**: Ensure same text + config always gets same cache key
2. **Embedding store/retrieve round-trip**: Data integrity
3. **Session creation respects min size**: Core algorithm correctness
4. **Coherence calculation**: Core metric correctness
5. **Import/export data integrity**: No data loss

### Should Have (P1)

1. **Batch operations efficiency**: Cache batch lookups work correctly
2. **Hybrid session handling**: Pre-assigned presentations respected
3. **Move presentation updates placements**: Data consistency
4. **Coverage query accuracy**: New method for EMB-001

### Nice to Have (P2)

1. **Truncation behavior**: Ollama context length handling
2. **Export format validation**: Bundle files have correct structure
3. **Error handling**: Graceful failures with clear messages

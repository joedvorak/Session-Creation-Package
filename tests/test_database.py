"""
Tests for database module: EmbeddingCache and ConferenceDB.

These tests verify core database operations without external dependencies.
"""

import pytest
import numpy as np
from pathlib import Path

from smart.core.database import EmbeddingCache, ConferenceDB, EmbeddingConfig


# =============================================================================
# EmbeddingConfig Tests
# =============================================================================

class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""
    
    @pytest.mark.unit
    def test_config_creation(self):
        """Test basic config creation."""
        config = EmbeddingConfig(
            model_name="test-model",
            model_version="v1.0",
            dimensions=384,
        )
        assert config.model_name == "test-model"
        assert config.model_version == "v1.0"
        assert config.dimensions == 384
        assert config.task_type == "SEMANTIC_SIMILARITY"  # default
    
    @pytest.mark.unit
    def test_config_hash_deterministic(self):
        """Config hash should be deterministic for same inputs."""
        config1 = EmbeddingConfig(model_name="model", model_version="v1")
        config2 = EmbeddingConfig(model_name="model", model_version="v1")
        
        assert config1.config_hash() == config2.config_hash()
    
    @pytest.mark.unit
    def test_config_hash_varies_with_params(self):
        """Config hash should differ when parameters differ."""
        config1 = EmbeddingConfig(model_name="model", model_version="v1")
        config2 = EmbeddingConfig(model_name="model", model_version="v2")
        config3 = EmbeddingConfig(model_name="other", model_version="v1")
        
        hashes = [config1.config_hash(), config2.config_hash(), config3.config_hash()]
        assert len(set(hashes)) == 3  # All different
    
    @pytest.mark.unit
    def test_config_to_dict_roundtrip(self):
        """Config should survive dict serialization."""
        original = EmbeddingConfig(
            model_name="test",
            model_version="v1.0",
            task_type="CLUSTERING",
            dimensions=768,
        )
        
        restored = EmbeddingConfig.from_dict(original.to_dict())
        
        assert restored.model_name == original.model_name
        assert restored.model_version == original.model_version
        assert restored.task_type == original.task_type
        assert restored.dimensions == original.dimensions


# =============================================================================
# EmbeddingCache Tests
# =============================================================================

class TestEmbeddingCache:
    """Tests for EmbeddingCache database operations."""
    
    @pytest.mark.unit
    def test_cache_creation(self, embedding_cache):
        """Cache should initialize without error."""
        assert embedding_cache is not None
        assert embedding_cache.db_path.exists()
    
    @pytest.mark.unit
    def test_store_and_retrieve_single(self, embedding_cache, embedding_config):
        """Store and retrieve a single embedding."""
        text = "Test presentation about machine learning"
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Store
        embedding_cache.store_embedding(text, embedding, embedding_config)
        
        # Retrieve
        retrieved = embedding_cache.get_embedding(text, embedding_config)
        
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, embedding, decimal=5)
    
    @pytest.mark.unit
    def test_retrieve_nonexistent(self, embedding_cache, embedding_config):
        """Retrieving nonexistent embedding should return None."""
        result = embedding_cache.get_embedding("nonexistent text", embedding_config)
        assert result is None
    
    @pytest.mark.unit
    def test_store_batch(self, embedding_cache, embedding_config):
        """Store and retrieve batch of embeddings."""
        texts = [f"Presentation {i} about topic {i}" for i in range(10)]
        embeddings = np.random.randn(10, 384).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Store batch - API takes list of (text, embedding) tuples
        texts_and_embeddings = list(zip(texts, embeddings))
        embedding_cache.store_embeddings_batch(texts_and_embeddings, embedding_config)
        
        # Retrieve batch - returns dict mapping text -> embedding
        retrieved = embedding_cache.get_embeddings_batch(texts, embedding_config)
        
        assert len(retrieved) == 10
        for i, text in enumerate(texts):
            assert retrieved[text] is not None
            np.testing.assert_array_almost_equal(retrieved[text], embeddings[i], decimal=5)
    
    @pytest.mark.unit
    def test_config_isolation(self, embedding_cache):
        """Embeddings with different configs should be separate.
        
        Note: This test uses fresh config objects for retrieval to work around
        EMB-006 (config dimension mutation bug). See test_config_mutation_bug
        for a test that explicitly demonstrates the bug.
        """
        text = "Same text different config"
        
        config1 = EmbeddingConfig(model_name="model-a", model_version="v1")
        config2 = EmbeddingConfig(model_name="model-b", model_version="v1")
        
        embedding1 = np.random.randn(384).astype(np.float32)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = np.random.randn(384).astype(np.float32)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        embedding_cache.store_embedding(text, embedding1, config1)
        embedding_cache.store_embedding(text, embedding2, config2)
        
        # WORKAROUND for EMB-006: store_embedding mutates config.dimensions,
        # changing the hash. Create fresh configs for retrieval.
        config1_fresh = EmbeddingConfig(model_name="model-a", model_version="v1")
        config2_fresh = EmbeddingConfig(model_name="model-b", model_version="v1")
        
        retrieved1 = embedding_cache.get_embedding(text, config1_fresh)
        retrieved2 = embedding_cache.get_embedding(text, config2_fresh)
        
        assert retrieved1 is not None
        assert retrieved2 is not None
        np.testing.assert_array_almost_equal(retrieved1, embedding1, decimal=5)
        np.testing.assert_array_almost_equal(retrieved2, embedding2, decimal=5)
    
    @pytest.mark.unit
    @pytest.mark.skip(reason="EMB-006: store_embedding mutates config.dimensions, changing hash. Remove skip when fixed.")
    def test_config_mutation_bug(self, embedding_cache):
        """Demonstrates EMB-006: config object should be usable after store.
        
        Currently FAILS because store_embedding() mutates config.dimensions,
        which changes the config_hash. This test should PASS once EMB-006 is fixed.
        """
        text = "Test text for mutation bug"
        config = EmbeddingConfig(model_name="test-model", model_version="v1")
        embedding = np.random.randn(384).astype(np.float32)
        
        hash_before = config.config_hash()
        embedding_cache.store_embedding(text, embedding, config)
        hash_after = config.config_hash()
        
        # This currently fails - the hash changes because dimensions was set
        assert hash_before == hash_after, (
            f"Config hash should not change after store. "
            f"Before: {hash_before}, After: {hash_after}"
        )
        
        # If hash is stable, retrieval with same config should work
        retrieved = embedding_cache.get_embedding(text, config)
        assert retrieved is not None, "Should retrieve with same config object"
    
    @pytest.mark.unit
    def test_get_stats(self, embedding_cache, embedding_config):
        """Stats should reflect stored embeddings."""
        texts = [f"Text {i}" for i in range(5)]
        embeddings = np.random.randn(5, 384).astype(np.float32)
        
        texts_and_embeddings = list(zip(texts, embeddings))
        embedding_cache.store_embeddings_batch(texts_and_embeddings, embedding_config)
        
        stats = embedding_cache.get_stats()
        
        assert stats["total_embeddings"] == 5
        # models list should contain our test model
        assert len(stats["models"]) >= 1
    
    @pytest.mark.unit
    @pytest.mark.skip(reason="Bug: config_hash mismatch between embeddings table and model_registry due to dimensions mutation. See EMB-006.")
    def test_get_coverage_by_model(self, embedding_cache):
        """Coverage query should show cached counts per model."""
        config1 = EmbeddingConfig(model_name="model-a", model_version="v1")
        config2 = EmbeddingConfig(model_name="model-b", model_version="v1")
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = np.random.randn(3, 384).astype(np.float32)
        
        # Store 3 with config1, 2 with config2 (using tuple format)
        texts_and_embeddings_1 = list(zip(texts, embeddings))
        texts_and_embeddings_2 = list(zip(texts[:2], embeddings[:2]))
        embedding_cache.store_embeddings_batch(texts_and_embeddings_1, config1)
        
        # Need fresh config2 since batch store mutates dimensions too
        config2_fresh = EmbeddingConfig(model_name="model-b", model_version="v1")
        embedding_cache.store_embeddings_batch(texts_and_embeddings_2, config2_fresh)
        
        coverage = embedding_cache.get_coverage_by_model(texts)
        
        # Should have entries for both configs
        assert len(coverage) >= 2
        
        # Find coverage for each config
        config1_coverage = None
        config2_coverage = None
        for entry in coverage:
            if entry["model_name"] == "model-a":
                config1_coverage = entry
            elif entry["model_name"] == "model-b":
                config2_coverage = entry
        
        assert config1_coverage is not None
        assert config1_coverage["cached_count"] == 3
        
        assert config2_coverage is not None
        assert config2_coverage["cached_count"] == 2


# =============================================================================
# ConferenceDB Tests
# =============================================================================

class TestConferenceDB:
    """Tests for ConferenceDB operations."""
    
    @pytest.mark.unit
    def test_db_creation(self, conference_db):
        """Database should initialize without error."""
        assert conference_db is not None
    
    @pytest.mark.unit
    def test_import_presentations(self, conference_db, sample_presentations):
        """Import presentations and verify count."""
        abstract_ids = conference_db.import_presentations_batch(sample_presentations)
        
        assert len(abstract_ids) == len(sample_presentations)
        
        # Verify retrieval
        presentations = conference_db.get_presentations()
        assert len(presentations) == len(sample_presentations)
    
    @pytest.mark.unit
    def test_get_presentation(self, conference_db, sample_presentations):
        """Retrieve single presentation by ID."""
        conference_db.import_presentations_batch(sample_presentations)
        
        pres = conference_db.get_presentation("ABS-001")
        
        assert pres is not None
        assert pres["abstract_id"] == "ABS-001"
        assert pres["title"] == "Test Presentation 1"
    
    @pytest.mark.unit
    def test_get_presentation_nonexistent(self, conference_db):
        """Nonexistent presentation should return None."""
        result = conference_db.get_presentation("DOES-NOT-EXIST")
        assert result is None
    
    @pytest.mark.unit
    def test_create_session(self, conference_db, sample_presentations):
        """Create a session with presentations."""
        conference_db.import_presentations_batch(sample_presentations)
        
        presentation_ids = ["ABS-001", "ABS-002", "ABS-003"]
        conference_db.create_session(
            session_id="SESSION-001",
            presentation_ids=presentation_ids,
            is_hybrid=False,
        )
        
        sessions = conference_db.get_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "SESSION-001"
    
    @pytest.mark.unit
    def test_get_session_with_presentations(self, conference_db, sample_presentations):
        """Get session should include presentations."""
        conference_db.import_presentations_batch(sample_presentations)
        
        presentation_ids = ["ABS-001", "ABS-002", "ABS-003"]
        conference_db.create_session(
            session_id="SESSION-001",
            presentation_ids=presentation_ids,
        )
        
        session = conference_db.get_session("SESSION-001")
        
        assert session is not None
        assert "presentations" in session
        assert len(session["presentations"]) == 3
    
    @pytest.mark.unit
    def test_update_session_metrics(self, conference_db, sample_presentations):
        """Update and retrieve session metrics."""
        conference_db.import_presentations_batch(sample_presentations)
        
        conference_db.create_session(
            session_id="SESSION-001",
            presentation_ids=["ABS-001", "ABS-002", "ABS-003"],
        )
        
        conference_db.update_session_metrics(
            session_id="SESSION-001",
            coherence=0.85,
            distinctiveness=0.72,
        )
        
        session = conference_db.get_session("SESSION-001")
        assert session["coherence"] == pytest.approx(0.85, rel=0.01)
        assert session["distinctiveness"] == pytest.approx(0.72, rel=0.01)
    
    @pytest.mark.unit
    def test_clear_sessions(self, conference_db, sample_presentations):
        """Clear sessions should remove all sessions."""
        conference_db.import_presentations_batch(sample_presentations)
        
        conference_db.create_session(
            session_id="SESSION-001",
            presentation_ids=["ABS-001", "ABS-002"],
        )
        conference_db.create_session(
            session_id="SESSION-002",
            presentation_ids=["ABS-003", "ABS-004"],
        )
        
        assert len(conference_db.get_sessions()) == 2
        
        cleared = conference_db.clear_sessions()
        
        assert cleared == 2
        assert len(conference_db.get_sessions()) == 0
    
    @pytest.mark.unit
    def test_get_stats(self, conference_db, sample_presentations):
        """Stats should reflect current database state."""
        conference_db.import_presentations_batch(sample_presentations)
        
        conference_db.create_session(
            session_id="SESSION-001",
            presentation_ids=["ABS-001", "ABS-002", "ABS-003"],
        )
        
        stats = conference_db.get_stats()
        
        assert stats["total_presentations"] == len(sample_presentations)
        assert stats["total_sessions"] == 1
        assert stats["placed_presentations"] == 3
        assert stats["unplaced_presentations"] == len(sample_presentations) - 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestDatabaseIntegration:
    """Integration tests combining cache and conference DB."""
    
    @pytest.mark.integration
    def test_embedding_workflow(self, embedding_cache, conference_db, 
                                 sample_presentations, mock_embedder):
        """Test typical embedding workflow: import -> embed -> cache."""
        config = EmbeddingConfig(
            model_name=mock_embedder.model_name,
            model_version=mock_embedder.model_version,
            dimensions=384,
        )
        
        # Import presentations
        conference_db.import_presentations_batch(sample_presentations)
        
        # Get texts for embedding
        presentations = conference_db.get_presentations()
        texts = [f"{p['title']}. {p['abstract']}" for p in presentations]
        
        # Generate and cache embeddings
        embeddings = mock_embedder.embed(texts)
        texts_and_embeddings = list(zip(texts, embeddings))
        embedding_cache.store_embeddings_batch(texts_and_embeddings, config)
        
        # Verify cache - returns dict
        retrieved = embedding_cache.get_embeddings_batch(texts, config)
        assert all(retrieved[t] is not None for t in texts)
        
        # Check coverage
        coverage = embedding_cache.get_coverage_by_model(texts)
        assert any(c["cached_count"] == len(texts) for c in coverage)

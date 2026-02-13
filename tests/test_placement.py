"""
Regression tests for placement algorithms.

TEST-006: Ensures placement behavior is stable and session distribution is balanced.

These tests catch:
- Determinism violations (same input should yield same output)
- Session size imbalance issues (PLACE-001: final filling overloads last sessions)
- Max session size violations (PLACE-002)
- Unassigned items when they should fit

Run with: pytest tests/test_placement.py -v
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from collections import Counter

# Add parent to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart.core.placement import (
    OralSessionPlacement,
    HybridFirstPlacement,
    SessionConstraints,
    PlacementResult,
)


# =============================================================================
# Test Fixtures - Placement Specific
# =============================================================================

@pytest.fixture
def clustered_embeddings() -> np.ndarray:
    """
    Create embeddings with clear cluster structure for predictable placement.
    
    Creates 5 clusters of 10 items each = 50 total items.
    This allows testing session formation with known topic structure.
    """
    np.random.seed(42)
    n_clusters = 5
    items_per_cluster = 10
    dimensions = 384
    
    # Create distinct cluster centers
    cluster_centers = []
    for i in range(n_clusters):
        # Space out centers in embedding space
        center = np.zeros(dimensions)
        center[i * 50:(i + 1) * 50] = 1.0  # Different "regions" active
        center = center / np.linalg.norm(center)
        cluster_centers.append(center)
    
    embeddings = []
    for cluster_idx in range(n_clusters):
        center = cluster_centers[cluster_idx]
        for _ in range(items_per_cluster):
            # Add small noise to cluster center
            noise = np.random.randn(dimensions) * 0.1
            embedding = center + noise
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
    
    return np.array(embeddings)


@pytest.fixture
def clustered_abstract_ids() -> List[str]:
    """Abstract IDs matching clustered_embeddings structure."""
    return [f"ABS-{i:03d}" for i in range(50)]


@pytest.fixture
def large_embeddings() -> np.ndarray:
    """
    Large embedding set for stress testing.
    
    120 items across 8 clusters - simulates real conference scale.
    """
    np.random.seed(123)
    n_clusters = 8
    items_per_cluster = 15
    dimensions = 384
    
    cluster_centers = np.random.randn(n_clusters, dimensions)
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    
    embeddings = []
    for cluster_idx in range(n_clusters):
        center = cluster_centers[cluster_idx]
        for _ in range(items_per_cluster):
            noise = np.random.randn(dimensions) * 0.15
            embedding = center + noise
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
    
    return np.array(embeddings)


@pytest.fixture
def large_abstract_ids() -> List[str]:
    """Abstract IDs for large embeddings (120 items)."""
    return [f"ABS-{i:04d}" for i in range(120)]


@pytest.fixture
def constraints_standard() -> SessionConstraints:
    """Standard session constraints: 8-12 items per session."""
    return SessionConstraints(
        min_session_size=8,
        max_session_size=12,
        max_sessions=None,
    )


@pytest.fixture
def constraints_tight() -> SessionConstraints:
    """Tight constraints: exactly 10 items per session."""
    return SessionConstraints(
        min_session_size=10,
        max_session_size=10,
        max_sessions=5,
    )


# =============================================================================
# Determinism Tests
# =============================================================================

@pytest.mark.unit
class TestPlacementDeterminism:
    """Verify placement algorithms produce consistent results."""
    
    def test_oral_placement_deterministic(
        self, clustered_embeddings, clustered_abstract_ids, constraints_standard
    ):
        """
        Same input should produce same output on multiple runs.
        
        This is critical for reproducibility and debugging.
        """
        strategy = OralSessionPlacement(
            linkage_method="average",
            tree_merge_stop=0.95,
        )
        
        # Run placement multiple times
        results = []
        for _ in range(3):
            result = strategy.place(
                embeddings=clustered_embeddings,
                abstract_ids=clustered_abstract_ids,
                constraints=constraints_standard,
            )
            results.append(result)
        
        # All runs should produce identical assignments
        for i in range(1, len(results)):
            assert results[0].session_assignments == results[i].session_assignments, \
                f"Run {i} produced different assignments than run 0"
            
            # Session structure should also match
            assert len(results[0].sessions) == len(results[i].sessions), \
                f"Run {i} produced different number of sessions"
    
    def test_oral_placement_deterministic_with_different_seed(
        self, clustered_abstract_ids, constraints_standard
    ):
        """
        Different random state shouldn't affect placement.
        
        The algorithm uses deterministic ordering based on linkage matrix.
        """
        # Create embeddings with different random states
        np.random.seed(42)
        emb1 = self._create_clustered_embeddings()
        
        np.random.seed(999)  # Different seed for numpy state
        # But use SAME embeddings
        emb2 = emb1.copy()
        
        strategy = OralSessionPlacement()
        
        result1 = strategy.place(emb1, clustered_abstract_ids, constraints_standard)
        result2 = strategy.place(emb2, clustered_abstract_ids, constraints_standard)
        
        assert result1.session_assignments == result2.session_assignments
    
    def _create_clustered_embeddings(self) -> np.ndarray:
        """Helper to create consistent test embeddings."""
        np.random.seed(42)  # Always same seed for consistency
        n_items = 50
        dimensions = 384
        embeddings = np.random.randn(n_items, dimensions)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings


# =============================================================================
# Session Balance Tests
# =============================================================================

@pytest.mark.unit
class TestSessionBalance:
    """
    Verify sessions are balanced and no session is grossly over/under-filled.
    
    These tests specifically target PLACE-001: the bug where final filling
    concentrates remaining items in the last few sessions.
    """
    
    @pytest.mark.xfail(reason="PLACE-002: _assign_remaining_items doesn't respect max_session_size in all cases")
    def test_no_session_exceeds_max_size(
        self, large_embeddings, large_abstract_ids, constraints_standard
    ):
        """
        PLACE-002: No session should exceed max_session_size.
        
        This is a hard constraint that must never be violated.
        
        KNOWN BUG: The final filling phase in _assign_remaining_items creates
        single-item overflow sessions that then get additional items assigned
        to them, potentially exceeding max_session_size.
        """
        strategy = OralSessionPlacement()
        result = strategy.place(
            embeddings=large_embeddings,
            abstract_ids=large_abstract_ids,
            constraints=constraints_standard,
        )
        
        for session in result.sessions:
            assert session["size"] <= constraints_standard.max_session_size, (
                f"Session {session['session_id']} has {session['size']} items, "
                f"exceeding max of {constraints_standard.max_session_size}"
            )
    
    def test_session_size_distribution_reasonable(
        self, large_embeddings, large_abstract_ids, constraints_standard
    ):
        """
        Session sizes should be relatively balanced.
        
        The standard deviation of session sizes should be small relative
        to the target size range.
        """
        strategy = OralSessionPlacement()
        result = strategy.place(
            embeddings=large_embeddings,
            abstract_ids=large_abstract_ids,
            constraints=constraints_standard,
        )
        
        sizes = [s["size"] for s in result.sessions]
        
        if len(sizes) > 1:
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            
            # CV (coefficient of variation) should be reasonable
            # With min=8, max=12, target ~10, CV should be < 0.3
            cv = std_size / mean_size if mean_size > 0 else 0
            
            assert cv < 0.4, (
                f"Session sizes too variable: CV={cv:.2f}, "
                f"sizes={sizes}"
            )
    
    def test_final_filling_distributes_evenly(
        self, large_embeddings, large_abstract_ids
    ):
        """
        PLACE-001 Regression: Final filling should not overload last sessions.
        
        When items remain after tree traversal, they should be distributed
        across ALL available sessions, not just fill the last ones.
        
        This test uses a scenario likely to trigger the bug:
        - Items that don't fit neatly into clusters
        - Settings that leave remainder items
        """
        # Use tight constraints to ensure remainder items exist
        constraints = SessionConstraints(
            min_session_size=10,
            max_session_size=12,
            max_sessions=10,
        )
        
        strategy = OralSessionPlacement(tree_merge_stop=0.8)  # Stop early to leave remainders
        result = strategy.place(
            embeddings=large_embeddings,
            abstract_ids=large_abstract_ids,
            constraints=constraints,
        )
        
        sizes = [s["size"] for s in result.sessions]
        
        # Check that no single session is more than 150% of average
        if len(sizes) > 1:
            mean_size = np.mean(sizes)
            max_size = max(sizes)
            
            # Known bug (PLACE-001): This assertion may fail until fixed
            # Currently marked as a known failure
            assert max_size <= mean_size * 1.5, (
                f"Session size imbalance detected: "
                f"max={max_size}, mean={mean_size:.1f}, "
                f"ratio={max_size/mean_size:.2f}. "
                f"Sizes: {sorted(sizes, reverse=True)}"
            )
    
    def test_no_tiny_sessions_after_placement(
        self, large_embeddings, large_abstract_ids, constraints_standard
    ):
        """
        Sessions should generally meet minimum size.
        
        Exception: The last session may be smaller if there aren't enough
        presentations to fill it.
        """
        strategy = OralSessionPlacement()
        result = strategy.place(
            embeddings=large_embeddings,
            abstract_ids=large_abstract_ids,
            constraints=constraints_standard,
        )
        
        # All but potentially one session should meet minimum
        sizes = sorted([s["size"] for s in result.sessions], reverse=True)
        
        underfilled = [s for s in sizes if s < constraints_standard.min_session_size]
        
        # At most one underfilled session is acceptable
        assert len(underfilled) <= 1, (
            f"Too many underfilled sessions: {underfilled}. "
            f"All sizes: {sizes}"
        )


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.unit
class TestPlacementEdgeCases:
    """Test placement behavior with edge cases."""
    
    def test_small_input_fewer_than_min_session(self):
        """Handle case where total items < min_session_size."""
        np.random.seed(42)
        embeddings = np.random.randn(5, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        abstract_ids = [f"ABS-{i:03d}" for i in range(5)]
        
        constraints = SessionConstraints(min_session_size=8, max_session_size=12)
        strategy = OralSessionPlacement()
        
        result = strategy.place(embeddings, abstract_ids, constraints)
        
        # Should still create one session with all items
        assert len(result.sessions) == 1
        assert result.sessions[0]["size"] == 5
    
    def test_single_item_input(self):
        """Handle single presentation."""
        np.random.seed(42)
        embeddings = np.random.randn(1, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        abstract_ids = ["ABS-001"]
        
        constraints = SessionConstraints(min_session_size=8, max_session_size=12)
        strategy = OralSessionPlacement()
        
        result = strategy.place(embeddings, abstract_ids, constraints)
        
        # Should create one session
        assert len(result.sessions) == 1
        assert "ABS-001" in result.session_assignments
    
    def test_exact_fit_sessions(self):
        """Items exactly fill sessions with no remainder."""
        np.random.seed(42)
        # 30 items with min=10 should create exactly 3 sessions
        embeddings = np.random.randn(30, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        abstract_ids = [f"ABS-{i:03d}" for i in range(30)]
        
        constraints = SessionConstraints(min_session_size=10, max_session_size=10)
        strategy = OralSessionPlacement()
        
        result = strategy.place(embeddings, abstract_ids, constraints)
        
        # All items should be assigned
        assert len(result.unassigned) == 0
        assert len(result.session_assignments) == 30


# =============================================================================
# Hybrid Placement Tests
# =============================================================================

@pytest.mark.unit
class TestHybridPlacement:
    """Tests specific to HybridFirstPlacement strategy."""
    
    def test_hybrid_sessions_filled_first(self, clustered_embeddings, clustered_abstract_ids):
        """Hybrid sessions should be filled before creating new sessions."""
        # Mark some items as pre-assigned to a hybrid session
        hybrid_assignments = {
            "ABS-000": "HYBRID-001",
            "ABS-001": "HYBRID-001",
        }
        
        constraints = SessionConstraints(min_session_size=8, max_session_size=12)
        strategy = HybridFirstPlacement()
        
        result = strategy.place(
            embeddings=clustered_embeddings,
            abstract_ids=clustered_abstract_ids,
            constraints=constraints,
            hybrid_assignments=hybrid_assignments,
        )
        
        # Hybrid session should exist and have been filled
        hybrid_session = None
        for s in result.sessions:
            if s["session_id"] == "HYBRID-001":
                hybrid_session = s
                break
        
        assert hybrid_session is not None, "Hybrid session not found in result"
        assert hybrid_session["size"] >= 2, "Hybrid session should have at least the pre-assigned items"
    
    def test_hybrid_preserves_assignments(self, clustered_embeddings, clustered_abstract_ids):
        """Pre-assigned items should remain in their original session."""
        hybrid_assignments = {
            "ABS-005": "HYBRID-SPECIAL",
            "ABS-010": "HYBRID-SPECIAL",
            "ABS-015": "HYBRID-SPECIAL",
        }
        
        constraints = SessionConstraints(min_session_size=8, max_session_size=12)
        strategy = HybridFirstPlacement()
        
        result = strategy.place(
            embeddings=clustered_embeddings,
            abstract_ids=clustered_abstract_ids,
            constraints=constraints,
            hybrid_assignments=hybrid_assignments,
        )
        
        # Verify pre-assigned items stayed in their session
        for abs_id, session_id in hybrid_assignments.items():
            assert result.session_assignments[abs_id] == session_id, \
                f"{abs_id} was moved from {session_id} to {result.session_assignments.get(abs_id)}"


# =============================================================================
# Similarity Tests
# =============================================================================

@pytest.mark.unit
class TestSessionCoherence:
    """Test that sessions contain similar items."""
    
    def test_clustered_items_placed_together(self, clustered_embeddings, clustered_abstract_ids, constraints_standard):
        """
        Items from the same cluster should generally be placed in the same session.
        
        Uses embeddings with known cluster structure.
        """
        strategy = OralSessionPlacement()
        result = strategy.place(
            embeddings=clustered_embeddings,
            abstract_ids=clustered_abstract_ids,
            constraints=constraints_standard,
        )
        
        # Check that items 0-9 (first cluster) mostly end up together
        first_cluster_ids = set(clustered_abstract_ids[:10])
        
        # Find which session has the most items from first cluster
        session_overlap = {}
        for session in result.sessions:
            overlap = len(set(session["presentation_ids"]) & first_cluster_ids)
            session_overlap[session["session_id"]] = overlap
        
        max_overlap = max(session_overlap.values())
        
        # At least 60% of first cluster should be in the same session
        assert max_overlap >= 6, (
            f"First cluster items too scattered: max overlap = {max_overlap}/10. "
            f"Session overlaps: {session_overlap}"
        )

    def test_all_clusters_maintain_cohesion(self, clustered_embeddings, clustered_abstract_ids, constraints_standard):
        """
        All 5 known clusters should maintain cohesion in their assigned sessions.
        
        This is a comprehensive test that verifies:
        1. Each of the 5 input clusters maps primarily to one session
        2. At least 60% of each cluster ends up in the same session
        3. Sessions don't heavily mix items from different clusters
        
        The clustered_embeddings fixture creates 5 clusters of 10 items each:
        - Cluster 0: items 0-9 (ABS-000 to ABS-009)
        - Cluster 1: items 10-19 (ABS-010 to ABS-019)
        - Cluster 2: items 20-29 (ABS-020 to ABS-029)
        - Cluster 3: items 30-39 (ABS-030 to ABS-039)
        - Cluster 4: items 40-49 (ABS-040 to ABS-049)
        """
        strategy = OralSessionPlacement()
        result = strategy.place(
            embeddings=clustered_embeddings,
            abstract_ids=clustered_abstract_ids,
            constraints=constraints_standard,
        )
        
        # Define the 5 known clusters based on fixture structure
        n_clusters = 5
        items_per_cluster = 10
        clusters = []
        for c in range(n_clusters):
            start_idx = c * items_per_cluster
            end_idx = start_idx + items_per_cluster
            cluster_ids = set(clustered_abstract_ids[start_idx:end_idx])
            clusters.append(cluster_ids)
        
        # Track which session each cluster primarily maps to
        cluster_to_best_session = {}
        cluster_cohesion_scores = {}
        
        for cluster_idx, cluster_ids in enumerate(clusters):
            # Find session with most items from this cluster
            best_session = None
            best_overlap = 0
            
            for session in result.sessions:
                session_ids = set(session["presentation_ids"])
                overlap = len(session_ids & cluster_ids)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_session = session["session_id"]
            
            cluster_to_best_session[cluster_idx] = best_session
            cohesion = best_overlap / len(cluster_ids)  # Fraction of cluster in best session
            cluster_cohesion_scores[cluster_idx] = cohesion
            
            # Each cluster should have at least 60% cohesion
            assert cohesion >= 0.6, (
                f"Cluster {cluster_idx} (items {cluster_idx * items_per_cluster}-"
                f"{(cluster_idx + 1) * items_per_cluster - 1}) has poor cohesion: "
                f"{best_overlap}/{len(cluster_ids)} = {cohesion:.1%} in session {best_session}"
            )
        
        # Verify clusters map to distinct sessions (no two clusters share same primary session)
        # This may not always hold if cluster size < min_session_size, but with 10 items
        # and min=8, it should generally work
        primary_sessions = list(cluster_to_best_session.values())
        unique_sessions = set(primary_sessions)
        
        # Allow some overlap but not complete collapse
        assert len(unique_sessions) >= 3, (
            f"Too many clusters collapsed into same sessions. "
            f"5 clusters mapped to only {len(unique_sessions)} sessions: {cluster_to_best_session}"
        )
        
    def test_session_purity(self, clustered_embeddings, clustered_abstract_ids, constraints_standard):
        """
        Sessions should not heavily mix items from different clusters.
        
        For each session, calculate "purity" = fraction from dominant cluster.
        Sessions should generally have >50% purity (more than half from one cluster).
        """
        strategy = OralSessionPlacement()
        result = strategy.place(
            embeddings=clustered_embeddings,
            abstract_ids=clustered_abstract_ids,
            constraints=constraints_standard,
        )
        
        # Define clusters
        n_clusters = 5
        items_per_cluster = 10
        
        def get_cluster_for_item(abs_id: str) -> int:
            """Return which cluster (0-4) an abstract ID belongs to."""
            idx = int(abs_id.split("-")[1])
            return idx // items_per_cluster
        
        low_purity_sessions = []
        
        for session in result.sessions:
            if session["size"] == 0:
                continue
                
            # Count items from each cluster in this session
            cluster_counts = Counter()
            for abs_id in session["presentation_ids"]:
                cluster_idx = get_cluster_for_item(abs_id)
                cluster_counts[cluster_idx] += 1
            
            # Purity = fraction from most common cluster
            dominant_count = cluster_counts.most_common(1)[0][1]
            purity = dominant_count / session["size"]
            
            if purity < 0.5:
                low_purity_sessions.append({
                    "session_id": session["session_id"],
                    "size": session["size"],
                    "purity": purity,
                    "cluster_distribution": dict(cluster_counts),
                })
        
        # Allow at most 1 low-purity session (edge cases from remainders)
        assert len(low_purity_sessions) <= 1, (
            f"Too many sessions with mixed clusters (purity < 50%): {low_purity_sessions}"
        )


# =============================================================================
# Performance / Stress Tests  
# =============================================================================

@pytest.mark.slow
class TestPlacementPerformance:
    """Performance tests for placement algorithms."""
    
    def test_scales_to_500_items(self):
        """Placement should handle 500 items reasonably quickly."""
        np.random.seed(42)
        n_items = 500
        embeddings = np.random.randn(n_items, 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        abstract_ids = [f"ABS-{i:04d}" for i in range(n_items)]
        
        constraints = SessionConstraints(min_session_size=8, max_session_size=12)
        strategy = OralSessionPlacement()
        
        import time
        start = time.time()
        result = strategy.place(embeddings, abstract_ids, constraints)
        elapsed = time.time() - start
        
        # Should complete in under 10 seconds
        assert elapsed < 10, f"Placement took {elapsed:.1f}s for {n_items} items"
        
        # Should have reasonable number of sessions
        expected_sessions = n_items // 10  # ~50 sessions
        assert len(result.sessions) >= expected_sessions * 0.8


# =============================================================================
# Metadata Tests
# =============================================================================

@pytest.mark.unit
class TestPlacementMetadata:
    """Verify placement result metadata is correct."""
    
    def test_metadata_item_counts(self, clustered_embeddings, clustered_abstract_ids, constraints_standard):
        """Metadata should accurately reflect item counts."""
        strategy = OralSessionPlacement()
        result = strategy.place(
            embeddings=clustered_embeddings,
            abstract_ids=clustered_abstract_ids,
            constraints=constraints_standard,
        )
        
        assert result.metadata["n_items"] == len(clustered_abstract_ids)
        assert result.metadata["n_assigned"] == len(result.session_assignments)
        assert result.metadata["n_unassigned"] == len(result.unassigned)
        
        # Verify counts are consistent
        total = result.metadata["n_assigned"] + result.metadata["n_unassigned"]
        assert total == result.metadata["n_items"]
    
    def test_result_structure(self, clustered_embeddings, clustered_abstract_ids, constraints_standard):
        """PlacementResult should have all expected fields."""
        strategy = OralSessionPlacement()
        result = strategy.place(
            embeddings=clustered_embeddings,
            abstract_ids=clustered_abstract_ids,
            constraints=constraints_standard,
        )
        
        # Check result type
        assert isinstance(result, PlacementResult)
        
        # Check required fields
        assert hasattr(result, "session_assignments")
        assert hasattr(result, "sessions")
        assert hasattr(result, "metadata")
        assert hasattr(result, "unassigned")
        
        # Check session structure
        for session in result.sessions:
            assert "session_id" in session
            assert "presentation_ids" in session
            assert "size" in session
            assert "is_hybrid" in session

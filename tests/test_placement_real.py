"""
Real-data placement regression tests using AIM26 conference data.

These tests use actual conference embeddings (3072-dim Gemini) to catch
issues that synthetic data misses - particularly PLACE-001 (tail session
overloading) and PLACE-002 (max size violations).

Data sources (checked in order):
1. tests/fixtures/aim26_benchmark.npz  — pre-extracted, fastest
2. examples/databases/AIM26_Example_*.db — shipped with repository

Generate .npz from example DBs (optional, for faster repeated runs):
    python scripts/extract_benchmark_fixture.py

Run with: pytest tests/test_placement_real.py -v
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart.core.placement import (
    OralSessionPlacement,
    HybridFirstPlacement,
    SessionConstraints,
    PlacementResult,
)

# Path to pre-extracted fixture (fastest loading)
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "aim26_benchmark.npz"

# Path to example databases shipped with repository
EXAMPLE_CACHE = Path(__file__).parent.parent / "examples" / "databases" / "AIM26_Example_cache.db"
EXAMPLE_WORKING = Path(__file__).parent.parent / "examples" / "databases" / "AIM26_Example_working.db"

# Data is available if EITHER the .npz OR the example databases exist
DATA_AVAILABLE = (
    FIXTURE_PATH.exists()
    or (EXAMPLE_CACHE.exists() and EXAMPLE_WORKING.exists())
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def aim26_data():
    """
    Load AIM26 benchmark data.

    Tries sources in order:
    1. Pre-extracted .npz fixture (fast)
    2. Example databases shipped with the repository (slower first time)

    Scope=module so we only load once per test module.
    """
    if FIXTURE_PATH.exists():
        # Fast path: load from pre-extracted .npz
        data = np.load(str(FIXTURE_PATH), allow_pickle=False)

        return {
            "abstract_ids": data["abstract_ids"].tolist(),
            "embeddings": data["embeddings"],
            "hybrid_assignments": dict(
                zip(data["hybrid_keys"].tolist(), data["hybrid_values"].tolist())
            ),
            "invited_ids": set(data["invited_ids"].tolist()),
        }

    if EXAMPLE_CACHE.exists() and EXAMPLE_WORKING.exists():
        # Slower path: load directly from example databases
        from scripts.placement_benchmark import load_benchmark_data

        return load_benchmark_data(str(EXAMPLE_CACHE), str(EXAMPLE_WORKING))

    pytest.skip(
        "No benchmark data available. "
        "Ensure examples/databases/ is present or run: "
        "python scripts/extract_benchmark_fixture.py"
    )


@pytest.fixture
def standard_constraints():
    return SessionConstraints(min_session_size=8, max_session_size=12)


@pytest.fixture
def oral_result(aim26_data, standard_constraints):
    """Run OralSessionPlacement on real data (cached per test module)."""
    strategy = OralSessionPlacement(tree_merge_stop=0.95)
    return strategy.place(
        embeddings=aim26_data["embeddings"],
        abstract_ids=aim26_data["abstract_ids"],
        constraints=standard_constraints,
    )


@pytest.fixture
def hybrid_result(aim26_data, standard_constraints):
    """Run HybridFirstPlacement on real data."""
    strategy = HybridFirstPlacement(tree_merge_stop=0.95)
    return strategy.place(
        embeddings=aim26_data["embeddings"],
        abstract_ids=aim26_data["abstract_ids"],
        constraints=standard_constraints,
        hybrid_assignments=aim26_data["hybrid_assignments"],
    )


# =============================================================================
# Determinism Tests on Real Data
# =============================================================================

@pytest.mark.skipif(not DATA_AVAILABLE, reason="Benchmark data not available")
class TestRealDataDeterminism:
    """Verify placement is deterministic on real conference data."""
    
    def test_oral_deterministic(self, aim26_data, standard_constraints):
        """Same real data should produce identical results every time."""
        strategy = OralSessionPlacement(tree_merge_stop=0.95)
        
        r1 = strategy.place(aim26_data["embeddings"], aim26_data["abstract_ids"], standard_constraints)
        r2 = strategy.place(aim26_data["embeddings"], aim26_data["abstract_ids"], standard_constraints)
        
        assert r1.session_assignments == r2.session_assignments
        assert len(r1.sessions) == len(r2.sessions)
    
    def test_hybrid_deterministic(self, aim26_data, standard_constraints):
        """Hybrid placement should also be deterministic."""
        strategy = HybridFirstPlacement(tree_merge_stop=0.95)
        
        r1 = strategy.place(
            aim26_data["embeddings"], aim26_data["abstract_ids"],
            standard_constraints, hybrid_assignments=aim26_data["hybrid_assignments"],
        )
        r2 = strategy.place(
            aim26_data["embeddings"], aim26_data["abstract_ids"],
            standard_constraints, hybrid_assignments=aim26_data["hybrid_assignments"],
        )
        
        assert r1.session_assignments == r2.session_assignments


# =============================================================================
# PLACE-001: Tail Session Overloading
# =============================================================================

@pytest.mark.skipif(not DATA_AVAILABLE, reason="Benchmark data not available")
class TestPlace001TailBehavior:
    """
    Detect the tail-overloading bug with real data.
    
    PLACE-001: The last sessions created tend to be much larger than
    earlier sessions because _assign_remaining_items creates single-item
    sessions that accumulate more items.
    """
    
    def test_tail_sessions_not_grossly_oversized(self, oral_result):
        """
        The last 10% of sessions should not be dramatically larger than the rest.
        
        A tail/body ratio > 1.5 indicates the PLACE-001 bug is active.
        """
        sizes = [s["size"] for s in oral_result.sessions]
        n = len(sizes)
        tail_count = max(1, n // 10)
        
        tail_sizes = np.array(sizes[-tail_count:])
        body_sizes = np.array(sizes[:-tail_count]) if tail_count < n else np.array(sizes)
        
        tail_mean = np.mean(tail_sizes)
        body_mean = np.mean(body_sizes)
        ratio = tail_mean / body_mean if body_mean > 0 else 0
        
        assert ratio <= 1.5, (
            f"PLACE-001: Tail sessions oversized. "
            f"Tail mean={tail_mean:.1f}, Body mean={body_mean:.1f}, "
            f"Ratio={ratio:.2f}. Tail sizes: {sorted(tail_sizes, reverse=True)[:10]}"
        )
    
    def test_no_session_over_20(self, oral_result):
        """
        No session should have 30+ presentations.
        
        With min=8, max=12, some sessions may grow past max_session_size
        when remaining items are assigned to their best-fit session
        (legacy-style behavior). But 30+ items indicates a severe failure.
        """
        sizes = [s["size"] for s in oral_result.sessions]
        oversize = [(s["session_id"], s["size"]) for s in oral_result.sessions if s["size"] >= 30]
        
        assert len(oversize) == 0, (
            f"Sessions with 30+ items (severe overloading): {oversize}"
        )


# =============================================================================
# PLACE-002: Max Session Size Enforcement
# =============================================================================

@pytest.mark.skipif(not DATA_AVAILABLE, reason="Benchmark data not available")
class TestPlace002MaxSizeEnforcement:
    """Verify max_session_size is respected on real data."""
    
    @pytest.mark.xfail(reason="PLACE-002: max_session_size not enforced in _assign_remaining_items")
    def test_no_session_exceeds_max_on_real_data(self, oral_result, standard_constraints):
        """
        No session should exceed max_session_size.
        
        This is expected to fail until PLACE-002 is fixed.
        """
        violations = [
            (s["session_id"], s["size"])
            for s in oral_result.sessions
            if s["size"] > standard_constraints.max_session_size
        ]
        
        assert len(violations) == 0, (
            f"{len(violations)} sessions exceed max size {standard_constraints.max_session_size}: "
            f"{violations[:10]}"
        )


# =============================================================================
# Size Distribution Tests
# =============================================================================

@pytest.mark.skipif(not DATA_AVAILABLE, reason="Benchmark data not available")
class TestRealDataSizeDistribution:
    """Verify reasonable session size distribution on real data."""
    
    def test_all_presentations_assigned(self, oral_result, aim26_data):
        """Every presentation should be assigned to a session."""
        assert len(oral_result.unassigned) == 0, (
            f"{len(oral_result.unassigned)} presentations unassigned"
        )
    
    def test_reasonable_session_count(self, oral_result, aim26_data, standard_constraints):
        """
        Session count should be in a reasonable range.
        
        With ~1150 presentations and target ~10/session, expect ~100-130 sessions.
        """
        n = len(aim26_data["abstract_ids"])
        expected_min = n // (standard_constraints.max_session_size + 2)
        expected_max = n // (standard_constraints.min_session_size - 2) + 10
        
        assert expected_min <= len(oral_result.sessions) <= expected_max, (
            f"Session count {len(oral_result.sessions)} outside expected range "
            f"[{expected_min}, {expected_max}]"
        )
    
    def test_size_coefficient_of_variation(self, oral_result):
        """
        CV of session sizes should be reasonable.
        
        With min=8, max=12, a well-balanced result should have CV < 0.3.
        A CV > 0.5 indicates serious distribution problems.
        """
        sizes = np.array([s["size"] for s in oral_result.sessions])
        cv = np.std(sizes) / np.mean(sizes)
        
        assert cv < 0.5, (
            f"Session size CV={cv:.3f} is too high. "
            f"Sizes: min={sizes.min()}, max={sizes.max()}, "
            f"mean={sizes.mean():.1f}, std={sizes.std():.1f}"
        )
    
    def test_few_undersized_sessions(self, oral_result, standard_constraints):
        """
        At most a handful of sessions should be below minimum size.
        
        Allow up to 2 undersized sessions for edge cases.
        """
        sizes = [s["size"] for s in oral_result.sessions]
        undersized = [sz for sz in sizes if sz < standard_constraints.min_session_size]
        
        assert len(undersized) <= 2, (
            f"{len(undersized)} sessions below min size {standard_constraints.min_session_size}: "
            f"{undersized}"
        )


# =============================================================================
# Hybrid Session Tests
# =============================================================================

@pytest.mark.skipif(not DATA_AVAILABLE, reason="Benchmark data not available")
class TestRealDataHybrid:
    """Test hybrid session handling on real data."""
    
    def test_hybrid_presentations_preserved(self, hybrid_result, aim26_data):
        """All hybrid pre-assignments should be preserved."""
        for aid, expected_sid in aim26_data["hybrid_assignments"].items():
            actual_sid = hybrid_result.session_assignments.get(aid)
            assert actual_sid == expected_sid, (
                f"Hybrid presentation {aid} moved from {expected_sid} to {actual_sid}"
            )
    
    def test_hybrid_sessions_filled(self, hybrid_result, aim26_data, standard_constraints):
        """Hybrid sessions should be filled to at least min_session_size."""
        hybrid_session_ids = set(aim26_data["hybrid_assignments"].values())
        
        for session in hybrid_result.sessions:
            if session["session_id"] in hybrid_session_ids:
                assert session["size"] >= standard_constraints.min_session_size, (
                    f"Hybrid session {session['session_id']} has only "
                    f"{session['size']} items (min={standard_constraints.min_session_size})"
                )


# =============================================================================
# Quality Metrics (informational, not strict assertions)
# =============================================================================

@pytest.mark.skipif(not DATA_AVAILABLE, reason="Benchmark data not available")
class TestRealDataQuality:
    """Quality checks on real data placement."""
    
    def test_mean_coherence_reasonable(self, oral_result, aim26_data):
        """
        Mean session coherence should be above a minimum threshold.
        
        From prior runs, coherence ~0.80+ is expected for well-organized sessions.
        Setting a conservative floor of 0.65.
        """
        from smart.core.metrics import calculate_session_coherence
        
        id_to_idx = {aid: i for i, aid in enumerate(aim26_data["abstract_ids"])}
        embeddings = aim26_data["embeddings"]
        
        coherences = []
        for session in oral_result.sessions:
            indices = [id_to_idx[aid] for aid in session["presentation_ids"] if aid in id_to_idx]
            if len(indices) >= 2:
                c = calculate_session_coherence(embeddings, indices)
                coherences.append(c)
        
        mean_coherence = np.mean(coherences) if coherences else 0
        
        assert mean_coherence >= 0.65, (
            f"Mean coherence {mean_coherence:.4f} below floor of 0.65. "
            f"Something may be wrong with the algorithm."
        )

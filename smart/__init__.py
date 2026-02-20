"""
SMART - Session Matching And Automated Recommendation Tool

A modular system for organizing conference sessions based on abstract similarity
using text embeddings and LLMs.
"""

__version__ = "2.0.0"

from smart.core.database import EmbeddingCache, ConferenceDB, EmbeddingConfig
from smart.core.placement import (
    PlacementStrategy, OralSessionPlacement, PosterThematicOrdering,
    SessionConstraints, PlacementResult, create_placement_strategy
)
from smart.core.metrics import (
    calculate_session_coherence, calculate_presentation_fit,
    calculate_all_metrics, find_outlier_presentations
)

__all__ = [
    # Database
    "EmbeddingCache",
    "ConferenceDB",
    "EmbeddingConfig",
    # Placement
    "PlacementStrategy",
    "OralSessionPlacement",
    "PosterThematicOrdering",
    "SessionConstraints",
    "PlacementResult",
    "create_placement_strategy",
    # Metrics
    "calculate_session_coherence",
    "calculate_presentation_fit",
    "calculate_all_metrics",
    "find_outlier_presentations",
]

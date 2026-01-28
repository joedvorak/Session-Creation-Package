"""
SMART Core Module

Contains database management, placement algorithms, and metrics calculations.
"""

from smart.core.database import EmbeddingCache, ConferenceDB, EmbeddingConfig
from smart.core.placement import (
    PlacementStrategy, OralSessionPlacement, PosterThematicOrdering,
    TraditionalClusterPlacement, SessionConstraints, PlacementResult,
    create_placement_strategy
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
    "TraditionalClusterPlacement",
    "SessionConstraints",
    "PlacementResult",
    "create_placement_strategy",
    # Metrics
    "calculate_session_coherence",
    "calculate_presentation_fit",
    "calculate_all_metrics",
    "find_outlier_presentations",
]

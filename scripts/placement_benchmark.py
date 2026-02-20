"""
Placement Benchmark Framework for SMART Session Organizer.

Provides:
- PlacementReport: Standardized metrics for any placement result
- BenchmarkComparison: Side-by-side comparison of multiple runs
- HTML report generation with embedded plots

Usage:
    from scripts.placement_benchmark import PlacementReport, BenchmarkComparison
    
    report = PlacementReport.from_placement_result(result, embeddings, abstract_ids)
    report.print_summary()
    
    comparison = BenchmarkComparison([report1, report2])
    comparison.print_table()
    comparison.save_html("output/benchmarks/comparison.html")
"""

import numpy as np
import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart.core.placement import PlacementResult, SessionConstraints
from smart.core.metrics import (
    calculate_session_coherence,
    calculate_session_distinctiveness,
    calculate_presentation_fit,
)


# =============================================================================
# PlacementReport: Standardized metrics for a single placement run
# =============================================================================

@dataclass
class SessionReport:
    """Metrics for a single session."""
    session_id: str
    size: int
    is_hybrid: bool
    coherence: float
    distinctiveness: float
    mean_fit: float
    std_fit: float
    presentation_ids: List[str]


@dataclass 
class PlacementReport:
    """
    Comprehensive metrics report for a placement run.
    
    Captures size distribution, coherence, distinctiveness, fit,
    and tail behavior for comparison and regression testing.
    """
    # Identity
    strategy_name: str
    label: str  # Human-readable label for comparison
    timestamp: str
    
    # Constraints used
    min_session_size: int
    max_session_size: int
    max_sessions: Optional[int]
    
    # Global counts
    n_presentations: int
    n_sessions: int
    n_hybrid_sessions: int
    n_unassigned: int
    
    # Size distribution
    sizes: List[int]
    size_mean: float
    size_std: float
    size_cv: float  # Coefficient of variation
    size_min: int
    size_max: int
    size_median: float
    size_histogram: Dict[int, int]  # size -> count
    
    # Balance metrics
    max_over_mean_ratio: float  # max_size / mean_size
    sessions_over_150pct: int  # Count of sessions >150% of mean
    sessions_under_min: int  # Count of sessions below min_session_size
    
    # Tail behavior (PLACE-001 detection)
    tail_session_count: int  # Number of sessions in last 10%
    tail_mean_size: float  # Mean size of last 10% of sessions
    tail_max_size: int
    body_mean_size: float  # Mean size of first 90% of sessions
    tail_to_body_ratio: float  # tail_mean / body_mean
    
    # Coherence
    coherence_values: List[float]
    coherence_mean: float
    coherence_std: float
    coherence_min: float
    coherence_max: float
    
    # Distinctiveness
    distinctiveness_values: List[float]
    distinctiveness_mean: float
    distinctiveness_std: float
    
    # Presentation fit
    fit_values: List[float]
    fit_mean: float
    fit_std: float
    fit_min: float
    
    # Per-session details
    session_reports: List[SessionReport]
    
    # Algorithm parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_placement_result(
        cls,
        result: PlacementResult,
        embeddings: np.ndarray,
        abstract_ids: List[str],
        constraints: SessionConstraints,
        label: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "PlacementReport":
        """
        Build a PlacementReport from a PlacementResult.
        
        Args:
            result: Output from a PlacementStrategy.place() call
            embeddings: The embeddings used for placement
            abstract_ids: List of abstract IDs matching embeddings
            constraints: Constraints used during placement
            label: Human-readable label for this run
            parameters: Algorithm parameters for tracking
        """
        id_to_idx = {aid: i for i, aid in enumerate(abstract_ids)}
        
        # Group by session
        session_to_ids: Dict[str, List[str]] = {}
        for aid, sid in result.session_assignments.items():
            if sid not in session_to_ids:
                session_to_ids[sid] = []
            session_to_ids[sid].append(aid)
        
        # Build per-session metrics
        session_reports = []
        all_session_embeddings = []  # For distinctiveness calculation
        session_indices_list = []
        
        for session in result.sessions:
            sid = session["session_id"]
            pres_ids = session.get("presentation_ids", session_to_ids.get(sid, []))
            indices = [id_to_idx[aid] for aid in pres_ids if aid in id_to_idx]
            
            session_indices_list.append(indices)
            if indices:
                all_session_embeddings.append(embeddings[indices])
            else:
                all_session_embeddings.append(np.empty((0, embeddings.shape[1])))
        
        coherence_values = []
        distinctiveness_values = []
        fit_values = []
        sizes = []
        
        for i, session in enumerate(result.sessions):
            sid = session["session_id"]
            indices = session_indices_list[i]
            is_hybrid = session.get("is_hybrid", False)
            size = len(indices)
            sizes.append(size)
            
            # Coherence
            if size >= 2:
                coherence = calculate_session_coherence(embeddings, indices)
            else:
                coherence = 1.0
            coherence_values.append(coherence)
            
            # Distinctiveness
            other_embs = [e for j, e in enumerate(all_session_embeddings) if j != i and len(e) > 0]
            if other_embs and size > 0:
                distinctiveness = calculate_session_distinctiveness(
                    all_session_embeddings[i], other_embs
                )
            else:
                distinctiveness = 1.0
            distinctiveness_values.append(distinctiveness)
            
            # Presentation fit
            session_fits = []
            for idx in indices:
                if size >= 2:
                    fit = calculate_presentation_fit(
                        embeddings[idx], embeddings[indices], exclude_self=True
                    )
                else:
                    fit = 1.0
                session_fits.append(fit)
                fit_values.append(fit)
            
            mean_fit = float(np.mean(session_fits)) if session_fits else 0.0
            std_fit = float(np.std(session_fits)) if len(session_fits) > 1 else 0.0
            
            session_reports.append(SessionReport(
                session_id=sid,
                size=size,
                is_hybrid=is_hybrid,
                coherence=coherence,
                distinctiveness=distinctiveness,
                mean_fit=mean_fit,
                std_fit=std_fit,
                presentation_ids=[abstract_ids[idx] for idx in indices],
            ))
        
        # Size distribution
        sizes_arr = np.array(sizes) if sizes else np.array([0])
        size_mean = float(np.mean(sizes_arr))
        size_std = float(np.std(sizes_arr))
        size_cv = size_std / size_mean if size_mean > 0 else 0.0
        
        unique, counts = np.unique(sizes_arr, return_counts=True)
        size_histogram = dict(zip(unique.tolist(), counts.tolist()))
        
        # Balance
        max_over_mean = float(np.max(sizes_arr) / size_mean) if size_mean > 0 else 0.0
        sessions_over_150 = int(np.sum(sizes_arr > size_mean * 1.5))
        sessions_under_min = int(np.sum(sizes_arr < constraints.min_session_size))
        
        # Tail behavior (PLACE-001)
        n_sessions = len(sizes)
        tail_count = max(1, n_sessions // 10)
        # Sessions are in creation order, so "tail" = last ones created
        tail_sizes = sizes_arr[-tail_count:]
        body_sizes = sizes_arr[:-tail_count] if tail_count < n_sessions else sizes_arr
        
        tail_mean = float(np.mean(tail_sizes)) if len(tail_sizes) > 0 else 0.0
        body_mean = float(np.mean(body_sizes)) if len(body_sizes) > 0 else 0.0
        tail_to_body = tail_mean / body_mean if body_mean > 0 else 0.0
        
        # Hybrid count
        n_hybrid = sum(1 for s in result.sessions if s.get("is_hybrid", False))
        
        strategy_name = result.metadata.get("strategy", label or "unknown")
        
        return cls(
            strategy_name=strategy_name,
            label=label or strategy_name,
            timestamp=datetime.now().isoformat(),
            min_session_size=constraints.min_session_size,
            max_session_size=constraints.max_session_size,
            max_sessions=constraints.max_sessions,
            n_presentations=len(result.session_assignments),
            n_sessions=n_sessions,
            n_hybrid_sessions=n_hybrid,
            n_unassigned=len(result.unassigned),
            sizes=sizes,
            size_mean=size_mean,
            size_std=size_std,
            size_cv=size_cv,
            size_min=int(np.min(sizes_arr)),
            size_max=int(np.max(sizes_arr)),
            size_median=float(np.median(sizes_arr)),
            size_histogram=size_histogram,
            max_over_mean_ratio=max_over_mean,
            sessions_over_150pct=sessions_over_150,
            sessions_under_min=sessions_under_min,
            tail_session_count=tail_count,
            tail_mean_size=tail_mean,
            tail_max_size=int(np.max(tail_sizes)) if len(tail_sizes) > 0 else 0,
            body_mean_size=body_mean,
            tail_to_body_ratio=tail_to_body,
            coherence_values=coherence_values,
            coherence_mean=float(np.mean(coherence_values)) if coherence_values else 0.0,
            coherence_std=float(np.std(coherence_values)) if coherence_values else 0.0,
            coherence_min=float(np.min(coherence_values)) if coherence_values else 0.0,
            coherence_max=float(np.max(coherence_values)) if coherence_values else 0.0,
            distinctiveness_values=distinctiveness_values,
            distinctiveness_mean=float(np.mean(distinctiveness_values)) if distinctiveness_values else 0.0,
            distinctiveness_std=float(np.std(distinctiveness_values)) if distinctiveness_values else 0.0,
            fit_values=fit_values,
            fit_mean=float(np.mean(fit_values)) if fit_values else 0.0,
            fit_std=float(np.std(fit_values)) if fit_values else 0.0,
            fit_min=float(np.min(fit_values)) if fit_values else 0.0,
            session_reports=session_reports,
            parameters=parameters or {},
        )
    
    def print_summary(self):
        """Print a compact summary to console."""
        print(f"\n{'=' * 60}")
        print(f"  Placement Report: {self.label}")
        print(f"  Strategy: {self.strategy_name}")
        print(f"  {self.timestamp}")
        print(f"{'=' * 60}")
        
        print(f"\n  Presentations: {self.n_presentations}  |  Sessions: {self.n_sessions}  |  Hybrid: {self.n_hybrid_sessions}  |  Unassigned: {self.n_unassigned}")
        print(f"  Constraints: min={self.min_session_size}, max={self.max_session_size}, max_sessions={self.max_sessions}")
        
        print(f"\n  SIZE DISTRIBUTION")
        print(f"  {'Mean':>8s}: {self.size_mean:6.1f}    {'Std':>8s}: {self.size_std:5.1f}    {'CV':>4s}: {self.size_cv:.3f}")
        print(f"  {'Min':>8s}: {self.size_min:6d}    {'Max':>8s}: {self.size_max:5d}    {'Median':>6s}: {self.size_median:.0f}")
        print(f"  Max/Mean ratio: {self.max_over_mean_ratio:.2f}    Over 150%: {self.sessions_over_150pct}    Under min: {self.sessions_under_min}")
        print(f"  Histogram: {self.size_histogram}")
        
        print(f"\n  TAIL BEHAVIOR (last {self.tail_session_count} sessions)")
        print(f"  Tail mean: {self.tail_mean_size:.1f}    Body mean: {self.body_mean_size:.1f}    Ratio: {self.tail_to_body_ratio:.2f}")
        print(f"  Tail max:  {self.tail_max_size}")
        flag = " *** IMBALANCED ***" if self.tail_to_body_ratio > 1.3 else ""
        print(f"  Status: {'BALANCED' if self.tail_to_body_ratio <= 1.3 else 'IMBALANCED'}{flag}")
        
        print(f"\n  COHERENCE")
        print(f"  Mean: {self.coherence_mean:.4f}    Std: {self.coherence_std:.4f}    Min: {self.coherence_min:.4f}    Max: {self.coherence_max:.4f}")
        
        print(f"\n  DISTINCTIVENESS")
        print(f"  Mean: {self.distinctiveness_mean:.4f}    Std: {self.distinctiveness_std:.4f}")
        
        print(f"\n  PRESENTATION FIT")
        print(f"  Mean: {self.fit_mean:.4f}    Std: {self.fit_std:.4f}    Min: {self.fit_min:.4f}")
        
        if self.parameters:
            print(f"\n  PARAMETERS")
            for k, v in self.parameters.items():
                print(f"  {k}: {v}")
        
        print(f"{'=' * 60}\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        d = {
            "strategy_name": self.strategy_name,
            "label": self.label,
            "timestamp": self.timestamp,
            "constraints": {
                "min_session_size": self.min_session_size,
                "max_session_size": self.max_session_size,
                "max_sessions": self.max_sessions,
            },
            "counts": {
                "presentations": self.n_presentations,
                "sessions": self.n_sessions,
                "hybrid": self.n_hybrid_sessions,
                "unassigned": self.n_unassigned,
            },
            "size": {
                "mean": self.size_mean,
                "std": self.size_std,
                "cv": self.size_cv,
                "min": self.size_min,
                "max": self.size_max,
                "median": self.size_median,
                "histogram": self.size_histogram,
            },
            "balance": {
                "max_over_mean_ratio": self.max_over_mean_ratio,
                "sessions_over_150pct": self.sessions_over_150pct,
                "sessions_under_min": self.sessions_under_min,
            },
            "tail": {
                "count": self.tail_session_count,
                "mean_size": self.tail_mean_size,
                "max_size": self.tail_max_size,
                "body_mean_size": self.body_mean_size,
                "ratio": self.tail_to_body_ratio,
            },
            "coherence": {
                "mean": self.coherence_mean,
                "std": self.coherence_std,
                "min": self.coherence_min,
                "max": self.coherence_max,
            },
            "distinctiveness": {
                "mean": self.distinctiveness_mean,
                "std": self.distinctiveness_std,
            },
            "fit": {
                "mean": self.fit_mean,
                "std": self.fit_std,
                "min": self.fit_min,
            },
            "parameters": self.parameters,
        }
        return d


# =============================================================================
# BenchmarkComparison: Side-by-side comparison
# =============================================================================

class BenchmarkComparison:
    """Compare multiple PlacementReport instances side by side."""
    
    def __init__(self, reports: List[PlacementReport]):
        self.reports = reports
    
    def print_table(self):
        """Print a comparison table to console."""
        if not self.reports:
            print("No reports to compare.")
            return
        
        labels = [r.label for r in self.reports]
        col_width = max(20, max(len(l) for l in labels) + 2)
        
        def row(metric_name, values, fmt=".1f"):
            vals = "  ".join(f"{v:{col_width}{fmt}}" if isinstance(v, float) 
                           else f"{str(v):>{col_width}s}" for v in values)
            print(f"  {metric_name:<28s}{vals}")
        
        header = "  ".join(f"{l:>{col_width}s}" for l in labels)
        print(f"\n{'=' * (30 + (col_width + 2) * len(labels))}")
        print(f"  BENCHMARK COMPARISON")
        print(f"{'=' * (30 + (col_width + 2) * len(labels))}")
        print(f"  {'Metric':<28s}{header}")
        print(f"  {'-' * 28}{'  '.join(['-' * col_width] * len(labels))}")
        
        # Counts
        row("Presentations", [r.n_presentations for r in self.reports], "d")
        row("Sessions", [r.n_sessions for r in self.reports], "d")
        row("Hybrid sessions", [r.n_hybrid_sessions for r in self.reports], "d")
        row("Unassigned", [r.n_unassigned for r in self.reports], "d")
        print()
        
        # Size
        row("Size mean", [r.size_mean for r in self.reports])
        row("Size std", [r.size_std for r in self.reports])
        row("Size CV", [r.size_cv for r in self.reports], ".3f")
        row("Size min", [r.size_min for r in self.reports], "d")
        row("Size max", [r.size_max for r in self.reports], "d")
        row("Size median", [r.size_median for r in self.reports])
        print()
        
        # Balance
        row("Max/Mean ratio", [r.max_over_mean_ratio for r in self.reports], ".2f")
        row("Sessions > 150% mean", [r.sessions_over_150pct for r in self.reports], "d")
        row("Sessions < min_size", [r.sessions_under_min for r in self.reports], "d")
        print()
        
        # Tail
        row("Tail mean size", [r.tail_mean_size for r in self.reports])
        row("Body mean size", [r.body_mean_size for r in self.reports])
        row("Tail/Body ratio", [r.tail_to_body_ratio for r in self.reports], ".2f")
        row("Tail max size", [r.tail_max_size for r in self.reports], "d")
        print()
        
        # Coherence
        row("Coherence mean", [r.coherence_mean for r in self.reports], ".4f")
        row("Coherence std", [r.coherence_std for r in self.reports], ".4f")
        row("Coherence min", [r.coherence_min for r in self.reports], ".4f")
        print()
        
        # Distinctiveness
        row("Distinctiveness mean", [r.distinctiveness_mean for r in self.reports], ".4f")
        row("Distinctiveness std", [r.distinctiveness_std for r in self.reports], ".4f")
        print()
        
        # Fit
        row("Fit mean", [r.fit_mean for r in self.reports], ".4f")
        row("Fit std", [r.fit_std for r in self.reports], ".4f")
        row("Fit min", [r.fit_min for r in self.reports], ".4f")
        
        print(f"{'=' * (30 + (col_width + 2) * len(labels))}\n")
    
    def generate_plots(self) -> List[Tuple[str, str]]:
        """
        Generate comparison plots as base64-encoded PNGs.
        
        Returns:
            List of (title, base64_png) tuples
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed. Skipping plots.")
            return []
        
        plots = []
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.reports), 1)))
        labels = [r.label for r in self.reports]
        
        # --- Plot 1: Session Size Distribution (histogram overlay) ---
        fig, ax = plt.subplots(figsize=(10, 5))
        all_sizes = []
        for r in self.reports:
            all_sizes.extend(r.sizes)
        if all_sizes:
            bin_min = min(all_sizes) - 1
            bin_max = max(all_sizes) + 2
            bins = np.arange(bin_min, bin_max + 1) - 0.5
        else:
            bins = np.arange(0, 20) - 0.5
        
        for i, r in enumerate(self.reports):
            ax.hist(r.sizes, bins=bins, alpha=0.5, label=r.label, 
                   color=colors[i], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel("Session Size")
        ax.set_ylabel("Count")
        ax.set_title("Session Size Distribution")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plots.append(("Session Size Distribution", _fig_to_base64(fig)))
        plt.close(fig)
        
        # --- Plot 2: Session Sizes by Creation Order ---
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, r in enumerate(self.reports):
            ax.plot(range(len(r.sizes)), r.sizes, 'o-', markersize=3, 
                   label=r.label, color=colors[i], alpha=0.7)
            ax.axhline(y=r.size_mean, color=colors[i], linestyle='--', alpha=0.4)
        
        ax.set_xlabel("Session Index (creation order)")
        ax.set_ylabel("Session Size")
        ax.set_title("Session Sizes by Creation Order (dashed = mean)")
        ax.legend()
        ax.grid(alpha=0.3)
        plots.append(("Session Sizes by Creation Order", _fig_to_base64(fig)))
        plt.close(fig)
        
        # --- Plot 3: Session Coherence by Creation Order ---
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, r in enumerate(self.reports):
            coherences = r.coherence_values
            ax.plot(range(len(coherences)), coherences, 'o-', markersize=3,
                   label=r.label, color=colors[i], alpha=0.7)
            ax.axhline(y=r.coherence_mean, color=colors[i], linestyle='--', alpha=0.4)
        
        ax.set_xlabel("Session Index (creation order)")
        ax.set_ylabel("Coherence")
        ax.set_title("Session Coherence by Creation Order (dashed = mean)")
        ax.legend()
        ax.grid(alpha=0.3)
        plots.append(("Session Coherence by Creation Order", _fig_to_base64(fig)))
        plt.close(fig)
        
        # --- Plot 4: Coherence Distribution (box plot) ---
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [r.coherence_values for r in self.reports]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i])
            box.set_alpha(0.5)
        ax.set_ylabel("Coherence")
        ax.set_title("Session Coherence Distribution")
        ax.grid(axis='y', alpha=0.3)
        plots.append(("Coherence Distribution", _fig_to_base64(fig)))
        plt.close(fig)
        
        # --- Plot 5: Coherence vs Session Size scatter ---
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, r in enumerate(self.reports):
            sizes = [sr.size for sr in r.session_reports]
            coherences = [sr.coherence for sr in r.session_reports]
            ax.scatter(sizes, coherences, alpha=0.5, label=r.label, 
                      color=colors[i], s=30)
        
        ax.set_xlabel("Session Size")
        ax.set_ylabel("Coherence")
        ax.set_title("Coherence vs Session Size")
        ax.legend()
        ax.grid(alpha=0.3)
        plots.append(("Coherence vs Size", _fig_to_base64(fig)))
        plt.close(fig)
        
        # --- Plot 6: Presentation Fit Distribution ---
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [r.fit_values for r in self.reports]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i])
            box.set_alpha(0.5)
        ax.set_ylabel("Presentation Fit")
        ax.set_title("Presentation-Session Fit Distribution")
        ax.grid(axis='y', alpha=0.3)
        plots.append(("Fit Distribution", _fig_to_base64(fig)))
        plt.close(fig)
        
        return plots
    
    def save_html(self, output_path: str):
        """
        Save a self-contained HTML report with embedded plots.
        
        This creates a single HTML file with all comparison data
        and plots embedded as base64 images. No external files needed.
        """
        plots = self.generate_plots()
        
        # Build comparison data for the table
        metrics_json = json.dumps([r.to_dict() for r in self.reports], indent=2)
        
        html = _build_html_report(self.reports, plots, metrics_json)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
        print(f"Report saved: {output_path}")
    
    def save_json(self, output_path: str):
        """Save comparison metrics as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "reports": [r.to_dict() for r in self.reports],
        }
        output_path.write_text(json.dumps(data, indent=2))
        print(f"JSON saved: {output_path}")


# =============================================================================
# Data Loading from Databases
# =============================================================================

def load_benchmark_data(
    cache_db_path: str,
    working_db_path: str,
) -> Dict[str, Any]:
    """
    Load embeddings and presentation data from SMART databases.
    
    Returns a dict with:
        - abstract_ids: List[str]
        - texts: List[str]  (combined_text for each presentation)
        - embeddings: np.ndarray (N x D)
        - config: EmbeddingConfig
        - hybrid_assignments: Dict[str, str]  (abstract_id -> session_id)
        - invited_ids: Set[str]  (abstract_ids of invited presentations)
    """
    import sqlite3
    
    # --- Load presentations from working DB ---
    conn_work = sqlite3.connect(working_db_path)
    c = conn_work.cursor()
    
    presentations = c.execute(
        'SELECT abstract_id, combined_text, is_invited FROM presentations ORDER BY abstract_id'
    ).fetchall()
    
    abstract_ids = [r[0] for r in presentations]
    texts = [r[1] for r in presentations]
    invited_ids = {r[0] for r in presentations if r[2]}
    
    # Load hybrid assignments from current placements
    hybrid_data = c.execute('''
        SELECT p.abstract_id, p.session_id 
        FROM placements p 
        JOIN sessions s ON p.session_id = s.session_id 
        WHERE s.is_hybrid = 1 AND p.is_current = 1
    ''').fetchall()
    hybrid_assignments = {r[0]: r[1] for r in hybrid_data}
    
    conn_work.close()
    
    # --- Load embeddings from cache DB ---
    conn_cache = sqlite3.connect(cache_db_path)
    c = conn_cache.cursor()
    
    # Get the model config
    model_info = c.execute(
        'SELECT DISTINCT model_name, model_version, task_type, dimensions FROM embeddings LIMIT 1'
    ).fetchone()
    
    from smart.core.database import EmbeddingConfig, _deserialize_embedding
    
    config = EmbeddingConfig(
        model_name=model_info[0],
        model_version=model_info[1],
        task_type=model_info[2],
        dimensions=model_info[3],
    )
    
    # Load embeddings matched to presentations by text hash
    import hashlib
    
    # Build text hash -> abstract_id mapping
    text_to_id = {}
    text_hashes = {}
    for aid, text in zip(abstract_ids, texts):
        h = hashlib.sha256(text.encode()).hexdigest()
        text_hashes[aid] = h
        text_to_id[h] = aid
    
    # Fetch all embeddings for this config
    config_hash = config.config_hash()
    rows = c.execute(
        'SELECT text_hash, embedding, dimensions FROM embeddings WHERE config_hash = ?',
        (config_hash,)
    ).fetchall()
    
    conn_cache.close()
    
    # Match embeddings to abstract_ids
    hash_to_embedding = {}
    for row in rows:
        emb = _deserialize_embedding(row[1], row[2])
        hash_to_embedding[row[0]] = emb
    
    # Build aligned arrays - only include presentations with embeddings
    matched_ids = []
    matched_embeddings = []
    matched_texts = []
    missing_ids = []
    
    for aid, text in zip(abstract_ids, texts):
        h = text_hashes[aid]
        if h in hash_to_embedding:
            matched_ids.append(aid)
            matched_embeddings.append(hash_to_embedding[h])
            matched_texts.append(text)
        else:
            missing_ids.append(aid)
    
    if missing_ids:
        print(f"Warning: {len(missing_ids)} presentations have no cached embedding")
    
    embeddings = np.array(matched_embeddings) if matched_embeddings else np.empty((0, config.dimensions))
    
    # Filter hybrid assignments to only matched IDs
    matched_set = set(matched_ids)
    hybrid_assignments = {k: v for k, v in hybrid_assignments.items() if k in matched_set}
    
    return {
        "abstract_ids": matched_ids,
        "texts": matched_texts,
        "embeddings": embeddings,
        "config": config,
        "hybrid_assignments": hybrid_assignments,
        "invited_ids": invited_ids & matched_set,
        "n_missing": len(missing_ids),
    }


# =============================================================================
# Legacy Adapter
# =============================================================================

def run_legacy_placement(
    embeddings: np.ndarray,
    abstract_ids: List[str],
    constraints: SessionConstraints,
    hybrid_assignments: Optional[Dict[str, str]] = None,
) -> PlacementResult:
    """
    Run the legacy create_sessions_w_hybrid() and convert to PlacementResult.
    
    This wraps the legacy function so it can be compared directly with
    current PlacementStrategy implementations.
    
    Args:
        embeddings: N x D array
        abstract_ids: List of abstract IDs matching embeddings
        constraints: Session constraints
        hybrid_assignments: Pre-existing session assignments
        
    Returns:
        PlacementResult in the same format as current strategies
    """
    import pandas as pd
    
    sys.path.insert(0, str(Path(__file__).parent.parent / "legacy"))
    from session_organizer import create_sessions_w_hybrid, COLUMNS
    
    # Index mapping
    id_to_idx = {aid: i for i, aid in enumerate(abstract_ids)}
    embedding_cols = [f"dim_{i}" for i in range(embeddings.shape[1])]
    
    # Prepare hybrid data if provided
    # Legacy expects df_presentations = GENERAL (non-hybrid) presentations only
    # and df_hybrid_presentations = the pre-assigned hybrid presentations separately
    df_hybrid = None
    hybrid_session_col = None
    df_hybrid_emb = None
    hybrid_ids_matched = []
    hybrid_id_set = set()
    
    if hybrid_assignments:
        hybrid_ids = list(hybrid_assignments.keys())
        hybrid_indices = [id_to_idx[aid] for aid in hybrid_ids if aid in id_to_idx]
        hybrid_ids_matched = [aid for aid in hybrid_ids if aid in id_to_idx]
        hybrid_sessions_matched = [hybrid_assignments[aid] for aid in hybrid_ids_matched]
        hybrid_id_set = set(hybrid_ids_matched)
        
        if hybrid_ids_matched:
            df_hybrid = pd.DataFrame({
                'Abstract ID': hybrid_ids_matched,
                'hybrid_session': hybrid_sessions_matched,
            })
            df_hybrid.index = range(len(hybrid_ids_matched))
            hybrid_session_col = 'hybrid_session'
            
            # Build hybrid embeddings
            hybrid_embs = embeddings[hybrid_indices]
            df_hybrid_emb = pd.DataFrame(hybrid_embs, columns=embedding_cols)
            df_hybrid_emb[COLUMNS['EMBEDDING_MODEL']] = "benchmark-model"
            df_hybrid_emb.index = df_hybrid.index
    
    # Build DataFrames for GENERAL presentations only (exclude hybrid)
    general_ids = [aid for aid in abstract_ids if aid not in hybrid_id_set]
    general_indices = [id_to_idx[aid] for aid in general_ids]
    general_embeddings = embeddings[general_indices] if general_indices else np.empty((0, embeddings.shape[1]))
    
    df_presentations = pd.DataFrame({
        'Abstract ID': general_ids,
        'Title': [f"Presentation {aid}" for aid in general_ids],
    })
    df_presentations.index = range(len(general_ids))
    
    df_embeddings = pd.DataFrame(general_embeddings, columns=embedding_cols)
    df_embeddings[COLUMNS['EMBEDDING_MODEL']] = "benchmark-model"
    df_embeddings.index = df_presentations.index
    
    # Check if hybrid sessions already have enough presentations
    # Legacy code can't handle sessions >= min_session_size (negative count_to_add)
    hybrid_skipped = False
    if df_hybrid is not None and hybrid_session_col:
        session_counts = df_hybrid.groupby(hybrid_session_col).size()
        if (session_counts >= constraints.min_session_size).any():
            # Can't fill sessions that are already full - run without hybrid
            # Add hybrid presentations back to the general pool
            all_ids = list(abstract_ids)
            all_embeddings = embeddings
            df_presentations = pd.DataFrame({
                'Abstract ID': all_ids,
                'Title': [f"Presentation {aid}" for aid in all_ids],
            })
            df_presentations.index = range(len(all_ids))
            df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_cols)
            df_embeddings[COLUMNS['EMBEDDING_MODEL']] = "benchmark-model"
            df_embeddings.index = df_presentations.index
            general_ids = all_ids  # Update for result mapping
            df_hybrid = None
            hybrid_session_col = None
            df_hybrid_emb = None
            hybrid_skipped = True
    
    # Run legacy placement
    df_sessions, labels, metadata = create_sessions_w_hybrid(
        df_presentations=df_presentations,
        similarity_func=cosine_similarity,
        df_presentation_embeddings=df_embeddings,
        df_hybrid_presentations=df_hybrid,
        hybrid_session_column=hybrid_session_col,
        df_hybrid_embeddings=df_hybrid_emb,
        max_sessions=constraints.max_sessions or 100,
        min_session_size=constraints.min_session_size,
        tree_merge_stop=0.95,
    )
    
    # Convert legacy output to PlacementResult
    session_assignments = {}
    sessions = []
    
    for _, row in df_sessions.iterrows():
        cluster_id = row[COLUMNS['CLUSTER_ID']]
        gen_indices = row[COLUMNS['GEN_PRESENTATION_INDICES']]
        hybrid_pres = row.get(COLUMNS['HYBRID_INVITED_PRESENTATIONS'], [])
        session_title = row.get(COLUMNS['FINAL_SESSION_TITLE'], '')
        
        session_id = f"LEGACY-{cluster_id:03d}"
        is_hybrid = bool(hybrid_pres and len(hybrid_pres) > 0)
        
        # Map gen_indices back to abstract_ids (these index into general_ids, not all abstract_ids)
        pres_ids = [general_ids[idx] for idx in gen_indices if idx < len(general_ids)]
        
        # Add hybrid presentation IDs  
        hybrid_pres_ids = []
        if hybrid_pres and len(hybrid_pres) > 0:
            # hybrid_pres contains DataFrame indices from the hybrid df
            hybrid_pres_ids = [hybrid_ids_matched[idx] for idx in hybrid_pres 
                              if idx < len(hybrid_ids_matched)]
            pres_ids = hybrid_pres_ids + pres_ids
        
        for aid in pres_ids:
            session_assignments[aid] = session_id
        
        sessions.append({
            "session_id": session_id,
            "is_hybrid": is_hybrid,
            "presentation_ids": pres_ids,
            "size": len(pres_ids),
        })
    
    unassigned = [aid for aid in abstract_ids if aid not in session_assignments]
    
    return PlacementResult(
        session_assignments=session_assignments,
        sessions=sessions,
        metadata={
            "strategy": "legacy_create_sessions_w_hybrid",
            "n_items": len(abstract_ids),
            "n_sessions": len(sessions),
            "n_assigned": len(session_assignments),
            "n_unassigned": len(unassigned),
            "hybrid_skipped": hybrid_skipped,
            "hybrid_skip_reason": "sessions already >= min_session_size" if hybrid_skipped else None,
        },
        unassigned=unassigned,
    )


# =============================================================================
# HTML Report Generation
# =============================================================================

def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _build_html_report(
    reports: List[PlacementReport], 
    plots: List[Tuple[str, str]],
    metrics_json: str,
) -> str:
    """Build self-contained HTML report."""
    
    # Build comparison table rows
    metrics = [
        ("Presentations", "n_presentations", "d"),
        ("Sessions", "n_sessions", "d"),
        ("Hybrid Sessions", "n_hybrid_sessions", "d"),
        ("Unassigned", "n_unassigned", "d"),
        ("", None, None),  # Separator
        ("Size Mean", "size_mean", ".1f"),
        ("Size Std", "size_std", ".1f"),
        ("Size CV", "size_cv", ".3f"),
        ("Size Min", "size_min", "d"),
        ("Size Max", "size_max", "d"),
        ("Size Median", "size_median", ".0f"),
        ("", None, None),
        ("Max/Mean Ratio", "max_over_mean_ratio", ".2f"),
        ("Sessions > 150% Mean", "sessions_over_150pct", "d"),
        ("Sessions < Min Size", "sessions_under_min", "d"),
        ("", None, None),
        ("Tail Mean Size", "tail_mean_size", ".1f"),
        ("Body Mean Size", "body_mean_size", ".1f"),
        ("Tail/Body Ratio", "tail_to_body_ratio", ".2f"),
        ("Tail Max Size", "tail_max_size", "d"),
        ("", None, None),
        ("Coherence Mean", "coherence_mean", ".4f"),
        ("Coherence Std", "coherence_std", ".4f"),
        ("Coherence Min", "coherence_min", ".4f"),
        ("", None, None),
        ("Distinctiveness Mean", "distinctiveness_mean", ".4f"),
        ("Distinctiveness Std", "distinctiveness_std", ".4f"),
        ("", None, None),
        ("Fit Mean", "fit_mean", ".4f"),
        ("Fit Std", "fit_std", ".4f"),
        ("Fit Min", "fit_min", ".4f"),
    ]
    
    def fmt(val, fmt_str):
        if fmt_str == "d":
            return str(int(val))
        return f"{val:{fmt_str}}"
    
    def highlight_best(values, attr, fmt_str):
        """Return list of (formatted_value, is_best) tuples."""
        # For most metrics, higher is not necessarily better
        # Green highlight rules:
        #   Smaller is better: size_std, size_cv, max_over_mean_ratio, sessions_over_150pct, 
        #                      sessions_under_min, tail_to_body_ratio, tail_max_size, n_unassigned
        #   Larger is better:  coherence_mean, distinctiveness_mean, fit_mean
        smaller_better = {
            "size_std", "size_cv", "max_over_mean_ratio", "sessions_over_150pct",
            "sessions_under_min", "tail_to_body_ratio", "tail_max_size", "n_unassigned",
            "coherence_std", "distinctiveness_std", "fit_std", "size_max", "tail_mean_size",
        }
        larger_better = {
            "coherence_mean", "coherence_min", "distinctiveness_mean", 
            "fit_mean", "fit_min", "size_min",
        }
        
        results = [(fmt(v, fmt_str), False) for v in values]
        
        if len(values) <= 1:
            return results
        
        if attr in smaller_better:
            best_idx = int(np.argmin(values))
        elif attr in larger_better:
            best_idx = int(np.argmax(values))
        else:
            return results
        
        results[best_idx] = (results[best_idx][0], True)
        return results
    
    # Build table HTML
    header_cells = "".join(f'<th>{r.label}</th>' for r in reports)
    
    table_rows = []
    for label, attr, fmt_str in metrics:
        if attr is None:
            table_rows.append('<tr class="sep"><td colspan="100"></td></tr>')
            continue
        
        values = [getattr(r, attr) for r in reports]
        formatted = highlight_best(values, attr, fmt_str)
        
        cells = "".join(
            f'<td class="{"best" if is_best else ""}">{val}</td>'
            for val, is_best in formatted
        )
        table_rows.append(f'<tr><td class="metric">{label}</td>{cells}</tr>')
    
    table_html = "\n".join(table_rows)
    
    # Build plot HTML
    plot_html_parts = []
    for title, b64 in plots:
        plot_html_parts.append(f'''
            <div class="plot">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{b64}" alt="{title}">
            </div>
        ''')
    plot_html = "\n".join(plot_html_parts)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SMART Placement Benchmark Report - {timestamp}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
        h1 {{ color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        h3 {{ color: #34495e; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
        th {{ background: #1a5276; color: white; padding: 10px 15px; text-align: right; }}
        th:first-child {{ text-align: left; }}
        td {{ padding: 6px 15px; text-align: right; border-bottom: 1px solid #eee; }}
        td.metric {{ text-align: left; font-weight: 500; }}
        td.best {{ background: #d4efdf; font-weight: bold; }}
        tr.sep td {{ padding: 3px; background: #f0f0f0; }}
        tr:hover {{ background: #f5f5f5; }}
        .plot {{ margin: 20px 0; text-align: center; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .params {{ background: #eef; padding: 10px 15px; border-radius: 4px;
                  margin: 5px 0; font-family: monospace; font-size: 0.9em; }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
        details {{ margin: 10px 0; }}
        summary {{ cursor: pointer; font-weight: 500; color: #1a5276; }}
    </style>
</head>
<body>
    <h1>SMART Placement Benchmark Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    
    <h2>Comparison Table</h2>
    <table>
        <thead>
            <tr><th>Metric</th>{header_cells}</tr>
        </thead>
        <tbody>
            {table_html}
        </tbody>
    </table>
    
    <h2>Parameters</h2>
    {"".join(f'<div class="params"><strong>{r.label}:</strong> {json.dumps(r.parameters)}</div>' for r in reports)}
    
    <h2>Plots</h2>
    {plot_html}
    
    <details>
        <summary>Raw Metrics (JSON)</summary>
        <pre>{metrics_json}</pre>
    </details>
</body>
</html>'''

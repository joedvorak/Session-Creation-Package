#!/usr/bin/env python3
"""
Run placement benchmarks comparing different strategies on real conference data.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --no-legacy          # Skip legacy comparison
    python scripts/run_benchmark.py --strategies oral     # Only run specific strategies
    python scripts/run_benchmark.py --output report.html  # Custom output path

Reads from:
    databases/AIM26-T2_cache.db   - Cached embeddings
    databases/AIM26-T2_working.db - Presentations and hybrid sessions

Outputs:
    output/benchmarks/<timestamp>_benchmark.html  - Self-contained HTML report
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from smart.core.placement import (
    OralSessionPlacement,
    HybridFirstPlacement,
    TraditionalClusterPlacement,
    SessionConstraints,
    create_placement_strategy,
)
from scripts.placement_benchmark import (
    PlacementReport,
    BenchmarkComparison,
    load_benchmark_data,
    run_legacy_placement,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run placement benchmarks")
    parser.add_argument(
        "--cache-db", default=str(PROJECT_ROOT / "databases" / "AIM26-T2_cache.db"),
        help="Path to embedding cache database",
    )
    parser.add_argument(
        "--working-db", default=str(PROJECT_ROOT / "databases" / "AIM26-T2_working.db"),
        help="Path to working database",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output HTML path (default: output/benchmarks/<timestamp>.html)",
    )
    parser.add_argument(
        "--no-legacy", action="store_true",
        help="Skip legacy create_sessions_w_hybrid comparison",
    )
    parser.add_argument(
        "--no-hybrid", action="store_true",
        help="Run without hybrid sessions (oral-only comparison)",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="Specific strategies to run (oral, hybrid_first, traditional, legacy)",
    )
    parser.add_argument(
        "--min-size", type=int, default=8,
        help="Minimum session size (default: 8)",
    )
    parser.add_argument(
        "--max-size", type=int, default=12,
        help="Maximum session size (default: 12)",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=None,
        help="Maximum number of sessions (default: auto)",
    )
    parser.add_argument(
        "--merge-stops", nargs="+", type=float, default=[0.95],
        help="tree_merge_stop values to test (default: 0.95)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Also save JSON metrics alongside HTML",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # --- Load data ---
    print("Loading benchmark data...")
    t0 = time.time()
    data = load_benchmark_data(args.cache_db, args.working_db)
    t_load = time.time() - t0
    
    embeddings = data["embeddings"]
    abstract_ids = data["abstract_ids"]
    hybrid_assignments = data["hybrid_assignments"] if not args.no_hybrid else {}
    
    print(f"  Loaded {len(abstract_ids)} presentations ({embeddings.shape[1]}D embeddings) in {t_load:.1f}s")
    print(f"  Hybrid sessions: {len(set(hybrid_assignments.values()))} ({len(hybrid_assignments)} presentations)")
    if data["n_missing"] > 0:
        print(f"  Warning: {data['n_missing']} presentations missing embeddings")
    
    # --- Configure ---
    constraints = SessionConstraints(
        min_session_size=args.min_size,
        max_session_size=args.max_size,
        max_sessions=args.max_sessions,
    )
    
    # Determine which strategies to run
    if args.strategies:
        strategy_names = args.strategies
    else:
        if hybrid_assignments:
            strategy_names = ["hybrid_first", "oral", "traditional"]
        else:
            strategy_names = ["oral", "traditional"]
        if not args.no_legacy:
            strategy_names.append("legacy")
    
    # --- Run benchmarks ---
    reports = []
    
    for merge_stop in args.merge_stops:
        for strat_name in strategy_names:
            if strat_name == "legacy":
                # Legacy adapter
                print(f"\nRunning legacy (merge_stop={merge_stop})...")
                t0 = time.time()
                try:
                    result = run_legacy_placement(
                        embeddings=embeddings,
                        abstract_ids=abstract_ids,
                        constraints=constraints,
                        hybrid_assignments=hybrid_assignments if hybrid_assignments else None,
                    )
                    elapsed = time.time() - t0
                    
                    label = f"Legacy (ms={merge_stop})"
                    params = {
                        "strategy": "legacy_create_sessions_w_hybrid",
                        "tree_merge_stop": merge_stop,
                        "elapsed_seconds": round(elapsed, 2),
                    }
                    
                    report = PlacementReport.from_placement_result(
                        result, embeddings, abstract_ids, constraints,
                        label=label, parameters=params,
                    )
                    reports.append(report)
                    print(f"  Done: {report.n_sessions} sessions in {elapsed:.1f}s")
                    
                except Exception as e:
                    print(f"  ERROR running legacy: {e}")
                    import traceback
                    traceback.print_exc()
                continue
            
            # Current strategies
            print(f"\nRunning {strat_name} (merge_stop={merge_stop})...")
            t0 = time.time()
            
            strategy = create_placement_strategy(
                strat_name,
                linkage_method="average",
                tree_merge_stop=merge_stop,
            )
            
            if strat_name == "hybrid_first" and hybrid_assignments:
                result = strategy.place(
                    embeddings=embeddings,
                    abstract_ids=abstract_ids,
                    constraints=constraints,
                    hybrid_assignments=hybrid_assignments,
                )
            else:
                result = strategy.place(
                    embeddings=embeddings,
                    abstract_ids=abstract_ids,
                    constraints=constraints,
                    hybrid_assignments=hybrid_assignments if hybrid_assignments else None,
                )
            
            elapsed = time.time() - t0
            
            suffix = f" (ms={merge_stop})" if len(args.merge_stops) > 1 else ""
            label = f"{strat_name}{suffix}"
            params = {
                "strategy": strat_name,
                "tree_merge_stop": merge_stop,
                "linkage_method": "average",
                "elapsed_seconds": round(elapsed, 2),
            }
            
            report = PlacementReport.from_placement_result(
                result, embeddings, abstract_ids, constraints,
                label=label, parameters=params,
            )
            reports.append(report)
            print(f"  Done: {report.n_sessions} sessions in {elapsed:.1f}s")
    
    # --- Print comparison ---
    comparison = BenchmarkComparison(reports)
    comparison.print_table()
    
    # --- Save outputs ---
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "output" / "benchmarks" / f"{timestamp}_benchmark.html"
    
    comparison.save_html(str(output_path))
    
    if args.json:
        json_path = output_path.with_suffix(".json")
        comparison.save_json(str(json_path))
    
    # --- Print individual summaries ---
    for report in reports:
        report.print_summary()
    
    print(f"\nBenchmark complete. {len(reports)} strategies compared.")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract a benchmark fixture from SMART databases for regression testing.

Creates a compact .npz file containing:
    - embeddings (N x D float32)
    - abstract_ids
    - hybrid_assignments
    - invited_ids

Usage:
    python scripts/extract_benchmark_fixture.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scripts.placement_benchmark import load_benchmark_data


def main():
    # Production databases (original, not committed to repo)
    prod_cache = PROJECT_ROOT / "databases" / "AIM26-T2_cache.db"
    prod_working = PROJECT_ROOT / "databases" / "AIM26-T2_working.db"

    # Example databases (deidentified, shipped with repo)
    example_cache = PROJECT_ROOT / "examples" / "databases" / "AIM26_Example_cache.db"
    example_working = PROJECT_ROOT / "examples" / "databases" / "AIM26_Example_working.db"

    # Prefer production databases, fall back to examples
    if prod_cache.exists() and prod_working.exists():
        cache_db = prod_cache
        working_db = prod_working
        print("Using production databases")
    elif example_cache.exists() and example_working.exists():
        cache_db = example_cache
        working_db = example_working
        print("Using example databases")
    else:
        print("ERROR: No databases found.")
        print(f"  Checked: {prod_cache}")
        print(f"  Checked: {example_cache}")
        sys.exit(1)

    output_dir = PROJECT_ROOT / "tests" / "fixtures"
    
    print("Loading data from databases...")
    data = load_benchmark_data(str(cache_db), str(working_db))
    
    embeddings = data["embeddings"]
    abstract_ids = data["abstract_ids"]
    hybrid_assignments = data["hybrid_assignments"]
    invited_ids = data["invited_ids"]
    
    print(f"  Presentations: {len(abstract_ids)}")
    print(f"  Embedding dims: {embeddings.shape[1]}")
    print(f"  Hybrid assignments: {len(hybrid_assignments)}")
    print(f"  Invited: {len(invited_ids)}")
    
    # Save as .npz (compressed)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / "aim26_benchmark.npz"
    
    np.savez_compressed(
        fixture_path,
        embeddings=embeddings.astype(np.float32),
        abstract_ids=np.array(abstract_ids, dtype=str),
        hybrid_keys=np.array(list(hybrid_assignments.keys()), dtype=str),
        hybrid_values=np.array(list(hybrid_assignments.values()), dtype=str),
        invited_ids=np.array(list(invited_ids), dtype=str),
    )
    
    file_size_mb = fixture_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {fixture_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()

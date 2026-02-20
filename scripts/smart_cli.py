#!/usr/bin/env python3
"""
SMART - Session Matching And Automated Recommendation Tool

Command-line script for organizing conference sessions based on abstract
similarity using text embeddings and LLMs.

This script demonstrates the complete SMART workflow:
1. Load presentations and hybrid sessions
2. Generate embeddings
3. Remove duplicates
4. Create sessions via hierarchical clustering
5. Calculate metrics
6. Generate titles with LLM
7. Match sessions to committees
8. Export results

Usage:
    python smart_cli.py [options]
    
Example:
    python smart_cli.py --presentations abstracts.xlsx --hybrid hybrid.xlsx --output output/

For help:
    python smart_cli.py --help
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load SMART library
from smart import (
    EmbeddingCache, ConferenceDB, EmbeddingConfig,
    SessionConstraints, PlacementResult,
    create_placement_strategy,
    calculate_all_metrics, find_outlier_presentations,
)
from smart.io.loaders import (
    load_presentations, load_hybrid_sessions, load_committees,
    ColumnMapping, inspect_file,
)
from smart.llm.embeddings import GeminiEmbedder, CachedEmbedder
from smart.llm.titles import GeminiTitleGenerator


# ============================================================
# DEFAULT CONFIGURATION
# Modify these values to match your data files
# ============================================================

DEFAULT_CONFIG = {
    # Conference settings
    "conference_year": 26,
    "conference_name": "AIM2026",
    
    # Input files
    "presentations_file": "abstracts 1.20.26.xlsx",
    "hybrid_sessions_file": "Hybrid Session Invited Presentations.xlsx",
    "committees_file": "ASABE Committees.csv",
    
    # Column mappings for main presentations file
    "presentations_columns": {
        "title": "Submission-Call for Abstracts-Presentation Title-Character max 160",
        "abstract": "Submission-Call for Abstracts-Abstract-Character max 4000-Abstracts will only be used to group into topical sessions and evaluate quality of talk.",
        "submission_id": "Submission-Call for Abstracts-Submission ID - 7 digits",
        "session_preference": "Submission-Call for Abstracts-Select Your Session Preference",
        "technical_community": "Submission-Call for Abstracts-Technical Community-First Preference ",
        "presenter_first_name": "Owner-First Name",
        "presenter_last_name": "Owner-Last Name",
        "presenter_email": "Owner-E-mail Address",
    },
    
    # Column mappings for hybrid sessions file
    "hybrid_columns": {
        "title": "Title",
        "abstract": "Abstract",
        "submission_id": "Submission ID - 7 digits",
        "session": "Session",
        "technical_community": "Technical Community",
        "presenter_first_name": "Presenter: First Name",
        "presenter_last_name": "Presenter: Last Name",
    },
    
    # Session filter
    "oral_preference_text": "Oral. I would like this submission to be considered for an Oral (standard or lightning) session.",
    
    # Session creation parameters
    "min_session_size": 9,
    "max_session_size": 12,
    "max_sessions": 111,
    
    # Models
    "embedding_model": "gemini-embedding-001",
    "title_model": "gemini-2.5-flash-lite",
    
    # Duplicate detection
    "duplicate_threshold": 0.99,
    
    # Output
    "output_dir": "output",
}


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_step(step: int, text: str):
    """Print a step indicator."""
    print(f"\n[Step {step}] {text}")
    print("-" * 40)


def load_env_and_validate():
    """Load environment variables and validate API key."""
    load_dotenv(".env")
    
    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: GEMINI_API_KEY not found in environment.")
        print("Please set it in your .env file:")
        print("  GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)


def run_smart_pipeline(config: dict, verbose: bool = True):
    """
    Run the complete SMART pipeline.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with results
    """
    results = {}
    
    # Setup
    print_header("SMART Session Organization Pipeline")
    print(f"Conference: {config['conference_name']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    # Initialize databases
    print_step(1, "Initializing databases and embedder")
    
    embedding_cache = EmbeddingCache(f"{config['conference_name']}_embeddings.db")
    conference_db = ConferenceDB(
        f"{config['conference_name']}_working.db",
        conference_year=config["conference_year"]
    )
    
    gemini_embedder = GeminiEmbedder(model=config["embedding_model"])
    embedder = CachedEmbedder(gemini_embedder, embedding_cache)
    
    cache_stats = embedding_cache.get_stats()
    print(f"✓ Embedding cache: {cache_stats['total_embeddings']} cached embeddings")
    print(f"✓ Conference database initialized")
    
    # Load hybrid sessions
    print_step(2, "Loading hybrid sessions (invited presentations)")
    
    df_hybrid = None
    df_hybrid_sessions = None
    hybrid_embeddings = None
    hybrid_abstract_ids = None
    
    hybrid_file = Path(config["hybrid_sessions_file"])
    if hybrid_file.exists():
        hybrid_mapping = ColumnMapping(
            title=config["hybrid_columns"]["title"],
            abstract=config["hybrid_columns"]["abstract"],
            submission_id=config["hybrid_columns"]["submission_id"],
            session=config["hybrid_columns"]["session"],
            technical_community=config["hybrid_columns"].get("technical_community"),
            presenter_first_name=config["hybrid_columns"].get("presenter_first_name"),
            presenter_last_name=config["hybrid_columns"].get("presenter_last_name"),
        )
        
        df_hybrid, df_hybrid_sessions, hybrid_metadata = load_hybrid_sessions(
            hybrid_file,
            hybrid_mapping,
            conference_year=config["conference_year"],
        )
        
        print(f"✓ Loaded {len(df_hybrid)} hybrid presentation slots")
        print(f"✓ Found {hybrid_metadata['session_count']} hybrid sessions")
        
        # Generate embeddings for hybrid
        print("\nGenerating embeddings for hybrid presentations...")
        hybrid_texts = df_hybrid["combined_text"].tolist()
        hybrid_embeddings_list = embedder.embed_batch(hybrid_texts, show_progress=verbose)
        hybrid_embeddings = np.array(hybrid_embeddings_list)
        hybrid_abstract_ids = df_hybrid["abstract_id"].tolist()
        
        # Import to database
        conference_db.import_presentations_batch(df_hybrid.to_dict('records'))
        print(f"✓ Embeddings generated: shape {hybrid_embeddings.shape}")
    else:
        print(f"No hybrid file found: {hybrid_file}")
        print("Proceeding without pre-assigned sessions...")
    
    # Load main presentations
    print_step(3, "Loading main presentations")
    
    pres_file = Path(config["presentations_file"])
    if not pres_file.exists():
        print(f"ERROR: Presentations file not found: {pres_file}")
        sys.exit(1)
    
    presentations_mapping = ColumnMapping(
        title=config["presentations_columns"]["title"],
        abstract=config["presentations_columns"]["abstract"],
        submission_id=config["presentations_columns"]["submission_id"],
        session_preference=config["presentations_columns"].get("session_preference"),
        technical_community=config["presentations_columns"].get("technical_community"),
        presenter_first_name=config["presentations_columns"].get("presenter_first_name"),
        presenter_last_name=config["presentations_columns"].get("presenter_last_name"),
        presenter_email=config["presentations_columns"].get("presenter_email"),
    )
    
    df_presentations, pres_metadata = load_presentations(
        pres_file,
        presentations_mapping,
        conference_year=config["conference_year"],
    )
    
    print(f"✓ Loaded {pres_metadata['loaded_count']} presentations")
    
    # Filter to oral presentations
    oral_pref = config.get("oral_preference_text")
    if oral_pref and "session_preference" in df_presentations.columns:
        df_oral = df_presentations[df_presentations["session_preference"] == oral_pref].copy()
        print(f"✓ Filtered to {len(df_oral)} oral presentations")
    else:
        df_oral = df_presentations.copy()
        print(f"Using all {len(df_oral)} presentations")
    
    # Generate embeddings
    print_step(4, "Generating embeddings for presentations")
    
    oral_texts = df_oral["combined_text"].tolist()
    print(f"Generating embeddings for {len(oral_texts)} presentations...")
    oral_embeddings_list = embedder.embed_batch(oral_texts, show_progress=verbose)
    oral_embeddings = np.array(oral_embeddings_list)
    oral_abstract_ids = df_oral["abstract_id"].tolist()
    
    print(f"✓ Embeddings generated: shape {oral_embeddings.shape}")
    
    # Remove duplicates
    print_step(5, "Removing near-duplicates")
    
    threshold = config["duplicate_threshold"]
    similarity_matrix = cosine_similarity(oral_embeddings, oral_embeddings)
    
    duplicates_to_remove = []
    n = len(oral_abstract_ids)
    
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                if verbose:
                    print(f"  Duplicate: {oral_abstract_ids[i]} ↔ {oral_abstract_ids[j]} (sim={similarity_matrix[i, j]:.4f})")
                duplicates_to_remove.append(j)
    
    duplicates_to_remove = sorted(set(duplicates_to_remove))
    print(f"Found {len(duplicates_to_remove)} near-duplicates to remove")
    
    if duplicates_to_remove:
        keep_mask = [i not in duplicates_to_remove for i in range(n)]
        df_oral = df_oral.iloc[keep_mask].reset_index(drop=True)
        oral_embeddings = oral_embeddings[keep_mask]
        oral_abstract_ids = [aid for i, aid in enumerate(oral_abstract_ids) if keep_mask[i]]
        print(f"✓ Remaining presentations: {len(df_oral)}")
    
    # Create sessions
    print_step(6, "Creating sessions via hierarchical clustering")
    
    hybrid_assignments = {}
    if df_hybrid is not None:
        for _, row in df_hybrid.iterrows():
            hybrid_assignments[row["abstract_id"]] = row["session_id"]
    
    print(f"Pre-assigned (hybrid): {len(hybrid_assignments)} presentations")
    
    # Choose strategy
    if hybrid_assignments:
        strategy = create_placement_strategy("hybrid_first")
    else:
        strategy = create_placement_strategy("oral")
    print(f"Using strategy: {strategy.name}")
    
    # Set constraints
    constraints = SessionConstraints(
        min_session_size=config["min_session_size"],
        max_session_size=config["max_session_size"],
        max_sessions=config["max_sessions"],
    )
    
    # Run placement
    place_kwargs = {
        "embeddings": oral_embeddings,
        "abstract_ids": oral_abstract_ids,
        "constraints": constraints,
        "hybrid_assignments": hybrid_assignments if hybrid_assignments else None,
    }
    
    if hybrid_assignments and hybrid_embeddings is not None:
        place_kwargs["hybrid_embeddings"] = dict(zip(hybrid_abstract_ids, hybrid_embeddings))
    
    result = strategy.place(**place_kwargs)
    
    print(f"✓ Created {len(result.sessions)} sessions")
    print(f"  - Assigned: {len(result.session_assignments)} presentations")
    print(f"  - Unassigned: {len(result.unassigned)} presentations")
    
    # Save to database
    df_oral["session_id"] = df_oral["abstract_id"].map(result.session_assignments)
    conference_db.import_presentations_batch(df_oral.to_dict('records'))
    
    for session in result.sessions:
        conference_db.create_session(
            session_id=session["session_id"],
            presentation_ids=session["presentation_ids"],
            is_hybrid=session.get("is_hybrid", False),
            placement_strategy=strategy.name,
        )
    
    # Calculate metrics
    print_step(7, "Calculating session metrics")
    
    metrics = calculate_all_metrics(
        oral_embeddings,
        oral_abstract_ids,
        result.session_assignments,
    )
    
    print(f"Mean Coherence: {metrics['summary']['mean_coherence']:.3f}")
    print(f"Mean Fit: {metrics['summary']['mean_fit']:.3f}")
    
    # Update database with metrics
    for session_id, sm in metrics["session_metrics"].items():
        conference_db.update_session_metrics(
            session_id,
            sm["coherence"],
            sm.get("distinctiveness")
        )
    
    # Build sessions DataFrame
    sessions_data = []
    for session_id, sm in metrics["session_metrics"].items():
        sessions_data.append({
            "session_id": session_id,
            "size": sm["size"],
            "coherence": sm["coherence"],
            "distinctiveness": sm.get("distinctiveness", 0),
            "presentation_ids": sm["presentation_ids"],
        })
    
    df_sessions = pd.DataFrame(sessions_data).sort_values("session_id").reset_index(drop=True)
    
    # Generate titles
    print_step(8, "Generating session titles and keywords")
    
    title_generator = GeminiTitleGenerator(model=config["title_model"])
    
    titles_list = []
    keywords_list = []
    
    for idx, row in df_sessions.iterrows():
        session_id = row["session_id"]
        pres_ids = row["presentation_ids"]
        
        # Get presentation details
        session_presentations = df_oral[df_oral["abstract_id"].isin(pres_ids)]
        pres_list = [
            {"title": r["title"], "abstract": r.get("abstract", "")}
            for _, r in session_presentations.iterrows()
        ]
        
        if verbose:
            print(f"  {session_id}: {len(pres_list)} presentations...", end=" ")
        
        try:
            gen_result = title_generator.generate(pres_list)
            titles_list.append(gen_result.titles)
            keywords_list.append(gen_result.keywords)
            
            conference_db.update_session_titles(
                session_id,
                gen_result.titles,
                gen_result.keywords,
                model_name=config["title_model"],
            )
            
            if verbose:
                print(f"✓ {gen_result.titles[0][:50]}..." if gen_result.titles else "✗")
        except Exception as e:
            print(f"Error: {e}")
            titles_list.append([])
            keywords_list.append([])
    
    df_sessions["title"] = [t[0] if t else "" for t in titles_list]
    df_sessions["title_options"] = titles_list
    df_sessions["keywords"] = [", ".join(k) for k in keywords_list]
    
    print(f"\n✓ Generated titles for {len(df_sessions)} sessions")
    
    # Committee matching (optional)
    committees_file = Path(config["committees_file"])
    if committees_file.exists():
        print_step(9, "Matching sessions to committees")
        
        df_committees, committee_metadata = load_committees(
            committees_file,
            name_column="Committee_Name",
            description_column="Description",
        )
        
        print(f"Loaded {committee_metadata['loaded_count']} committees")
        
        committee_texts = df_committees["combined_text"].tolist()
        committee_embeddings_list = embedder.embed_batch(committee_texts, show_progress=verbose)
        committee_embeddings = np.array(committee_embeddings_list)
        
        # Match sessions to committees
        session_matches = []
        for idx, row in df_sessions.iterrows():
            session_id = row["session_id"]
            pres_ids = row["presentation_ids"]
            
            pres_indices = [oral_abstract_ids.index(pid) for pid in pres_ids if pid in oral_abstract_ids]
            if not pres_indices:
                continue
            
            session_embs = oral_embeddings[pres_indices]
            session_centroid = np.mean(session_embs, axis=0, keepdims=True)
            similarities = cosine_similarity(session_centroid, committee_embeddings)[0]
            
            top_indices = np.argsort(similarities)[-3:][::-1]
            
            matches = []
            for rank, idx in enumerate(top_indices, 1):
                committee_id = df_committees.iloc[idx]["committee_id"]
                committee_name = df_committees.iloc[idx]["committee_name"]
                score = similarities[idx]
                matches.append((committee_id, committee_name, float(score), rank))
            
            if matches:
                conference_db.store_session_committee_matches(
                    session_id,
                    [(m[0], m[2], m[3]) for m in matches]
                )
            
            session_matches.append({
                "session_id": session_id,
                "committee_1": matches[0][1] if matches else "",
                "score_1": matches[0][2] if matches else 0,
            })
        
        df_committee_matches = pd.DataFrame(session_matches)
        df_sessions = df_sessions.merge(df_committee_matches, on="session_id", how="left")
        
        print(f"✓ Matched sessions to committees")
    
    # Export results
    print_step(10, "Exporting results")
    
    output_prefix = f"{output_dir}/{config['conference_name']}"
    
    # Export presentations
    df_export = conference_db.export_to_dataframe(flatten_extra_data=True)
    session_titles = dict(zip(df_sessions["session_id"], df_sessions["title"]))
    df_export["session_title"] = df_export["session_id"].map(session_titles)
    
    df_export.to_parquet(f"{output_prefix}_presentations.parquet", compression='snappy')
    print(f"✓ {output_prefix}_presentations.parquet")
    
    # Export sessions
    df_sessions.to_parquet(f"{output_prefix}_sessions.parquet", compression='snappy')
    df_sessions.to_csv(f"{output_prefix}_sessions.csv", index=False)
    print(f"✓ {output_prefix}_sessions.parquet")
    print(f"✓ {output_prefix}_sessions.csv")
    
    # Export similarity matrix
    pres_sim_matrix = cosine_similarity(oral_embeddings, oral_embeddings)
    df_pres_sim = pd.DataFrame(pres_sim_matrix, index=oral_abstract_ids, columns=oral_abstract_ids)
    df_pres_sim.to_parquet(f"{output_prefix}_pres_similarities.parquet", compression='snappy')
    print(f"✓ {output_prefix}_pres_similarities.parquet")
    
    # Redacted export (no PII)
    columns_to_redact = ["presenter_email", "presenter_first_name", "presenter_last_name", 
                         "affiliation", "abstract"]
    df_redacted = df_export.drop(columns=[c for c in columns_to_redact if c in df_export.columns], errors='ignore')
    df_redacted.to_parquet(f"{output_prefix}_presentations_public.parquet", compression='snappy')
    print(f"✓ {output_prefix}_presentations_public.parquet")
    
    # Summary
    print_header("PIPELINE COMPLETE")
    print(f"\nConference: {config['conference_name']}")
    print(f"\nPresentations:")
    print(f"  - Total loaded: {len(df_oral)}")
    print(f"  - Assigned: {len(result.session_assignments)}")
    print(f"  - Unassigned: {len(result.unassigned)}")
    print(f"\nSessions:")
    print(f"  - Total: {len(df_sessions)}")
    print(f"  - Mean coherence: {metrics['summary']['mean_coherence']:.3f}")
    print(f"  - Mean fit: {metrics['summary']['mean_fit']:.3f}")
    print(f"\nOutput files: {output_dir}/")
    
    cache_stats = embedding_cache.get_stats()
    print(f"\nEmbedding cache: {cache_stats['total_embeddings']} total embeddings")
    
    # Return results
    results = {
        "df_presentations": df_export,
        "df_sessions": df_sessions,
        "metrics": metrics,
        "placement_result": result,
        "config": config,
    }
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SMART - Session Matching And Automated Recommendation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--presentations", "-p",
        help="Path to main presentations file (CSV or Excel)",
        default=DEFAULT_CONFIG["presentations_file"]
    )
    parser.add_argument(
        "--hybrid", "-H",
        help="Path to hybrid sessions file (optional)",
        default=DEFAULT_CONFIG["hybrid_sessions_file"]
    )
    parser.add_argument(
        "--committees", "-c",
        help="Path to committees file (optional)",
        default=DEFAULT_CONFIG["committees_file"]
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory",
        default=DEFAULT_CONFIG["output_dir"]
    )
    parser.add_argument(
        "--name", "-n",
        help="Conference name (used for output file prefix)",
        default=DEFAULT_CONFIG["conference_name"]
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        help="Conference year (two digits)",
        default=DEFAULT_CONFIG["conference_year"]
    )
    parser.add_argument(
        "--min-size",
        type=int,
        help="Minimum session size",
        default=DEFAULT_CONFIG["min_session_size"]
    )
    parser.add_argument(
        "--max-size",
        type=int,
        help="Maximum session size",
        default=DEFAULT_CONFIG["max_session_size"]
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        help="Maximum number of sessions",
        default=DEFAULT_CONFIG["max_sessions"]
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Load and validate environment
    load_env_and_validate()
    
    # Build config from args and defaults
    config = DEFAULT_CONFIG.copy()
    config["presentations_file"] = args.presentations
    config["hybrid_sessions_file"] = args.hybrid
    config["committees_file"] = args.committees
    config["output_dir"] = args.output
    config["conference_name"] = args.name
    config["conference_year"] = args.year
    config["min_session_size"] = args.min_size
    config["max_session_size"] = args.max_size
    config["max_sessions"] = args.max_sessions
    
    # Run pipeline
    try:
        results = run_smart_pipeline(config, verbose=not args.quiet)
        print("\n✓ Pipeline completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Migration script for importing existing parquet/CSV data into SMART SQLite databases.

This script handles migration of:
- Presentations from existing CSV/parquet files
- Pre-computed embeddings
- Existing session assignments
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# Add parent to path for smart import
sys.path.insert(0, str(Path(__file__).parent))

from smart.core.database import EmbeddingCache, ConferenceDB, EmbeddingConfig


def migrate_presentations(
    source_file: Path,
    conference_db: ConferenceDB,
    column_mapping: Optional[Dict[str, str]] = None,
    conference_year: int = 26,
) -> Dict[str, Any]:
    """
    Migrate presentations from CSV/parquet to ConferenceDB.
    
    Args:
        source_file: Path to source file (CSV, Excel, or Parquet)
        conference_db: Target ConferenceDB instance
        column_mapping: Optional mapping of source columns to target fields
        conference_year: 2-digit year for ID generation
        
    Returns:
        Migration statistics
    """
    # Load source data
    ext = source_file.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(source_file)
    elif ext == ".csv":
        df = pd.read_csv(source_file)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(source_file)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    print(f"Loaded {len(df)} rows from {source_file}")
    print(f"Columns: {list(df.columns)}")
    
    # Default column mapping for common formats
    default_mapping = {
        # From existing session creation notebooks
        "Paper Title": "title",
        "Abstract": "abstract",
        "Submission ID": "submission_id",
        "abstract_id": "abstract_id",
        "combined_text": "combined_text",
        # Presenter info
        "Email": "presenter_email",
        "Presenter First Name": "presenter_first_name",
        "Presenter Last Name": "presenter_last_name",
        "First Name": "presenter_first_name",
        "Last Name": "presenter_last_name",
        # Session assignment
        "session_id": "session_id",
        "cluster": "session_id",
    }
    
    if column_mapping:
        default_mapping.update(column_mapping)
    
    # Map columns
    mapped_data = []
    temp_id_count = 0
    
    for _, row in df.iterrows():
        record = {}
        
        # Map each field
        for source_col, target_field in default_mapping.items():
            if source_col in df.columns:
                value = row.get(source_col)
                if pd.notna(value):
                    record[target_field] = value
        
        # Ensure required fields
        if "title" not in record:
            # Try fallbacks
            for fallback in ["Paper Title", "title", "Title", "TITLE"]:
                if fallback in row.index and pd.notna(row.get(fallback)):
                    record["title"] = row[fallback]
                    break
        
        if "title" not in record:
            print(f"  Skipping row without title: {row.to_dict()}")
            continue
        
        # Generate abstract_id if not present
        if "abstract_id" not in record:
            if "submission_id" in record:
                # Use submission ID to generate abstract_id
                sub_id = str(record["submission_id"])
                # Try to extract numeric portion
                numeric = "".join(c for c in sub_id if c.isdigit())
                if numeric:
                    record["abstract_id"] = f"{conference_year}{int(numeric):05d}"
                else:
                    temp_id_count += 1
                    record["abstract_id"] = f"TEMP-{conference_year}-{temp_id_count:05d}"
            else:
                temp_id_count += 1
                record["abstract_id"] = f"TEMP-{conference_year}-{temp_id_count:05d}"
        
        # Generate combined_text if not present
        if "combined_text" not in record:
            title = record.get("title", "")
            abstract = record.get("abstract", "")
            record["combined_text"] = f"{title}\n\n{abstract}".strip()
        
        mapped_data.append(record)
    
    # Import to database
    presentations = []
    for record in mapped_data:
        presentations.append({
            "abstract_id": record["abstract_id"],
            "title": record.get("title", ""),
            "abstract": record.get("abstract"),
            "submission_id": record.get("submission_id"),
            "combined_text": record.get("combined_text"),
            "presenter_email": record.get("presenter_email"),
            "presenter_first_name": record.get("presenter_first_name"),
            "presenter_last_name": record.get("presenter_last_name"),
        })
    
    imported_ids = conference_db.import_presentations_batch(presentations)
    
    # Handle session assignments if present
    session_count = 0
    if any("session_id" in r for r in mapped_data):
        session_assignments = {}
        for record in mapped_data:
            if "session_id" in record:
                sid = record["session_id"]
                if sid not in session_assignments:
                    session_assignments[sid] = []
                session_assignments[sid].append(record["abstract_id"])
        
        for session_id, pres_ids in session_assignments.items():
            conference_db.create_session(
                session_id=str(session_id),
                presentation_ids=pres_ids,
                placement_strategy="imported",
            )
            session_count += 1
    
    return {
        "total_rows": len(df),
        "imported_presentations": len(imported_ids),
        "temp_ids_generated": temp_id_count,
        "sessions_imported": session_count,
    }


def migrate_embeddings(
    embeddings_file: Path,
    embedding_cache: EmbeddingCache,
    texts_df: Optional[pd.DataFrame] = None,
    config: Optional[EmbeddingConfig] = None,
) -> Dict[str, Any]:
    """
    Migrate pre-computed embeddings to EmbeddingCache.
    
    Args:
        embeddings_file: Path to embeddings file (parquet or numpy)
        embedding_cache: Target cache instance
        texts_df: DataFrame with combined_text column (required if embeddings 
                  file doesn't include texts)
        config: EmbeddingConfig for these embeddings
        
    Returns:
        Migration statistics
    """
    # Default config for legacy embeddings
    if config is None:
        config = EmbeddingConfig(
            model_name="gemini-embedding-001",
            model_version="1.0",
            task_type="SEMANTIC_SIMILARITY",
            dimensions=768,
        )
    
    # Load embeddings
    ext = embeddings_file.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(embeddings_file)
        
        # Check for embedding column
        embedding_col = None
        for col in ["embedding", "embeddings", "vector"]:
            if col in df.columns:
                embedding_col = col
                break
        
        if embedding_col is None:
            # Look for array-like columns
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        first_val = df[col].iloc[0]
                        if isinstance(first_val, (list, np.ndarray)) and len(first_val) > 100:
                            embedding_col = col
                            break
                    except:
                        pass
        
        if embedding_col is None:
            raise ValueError(f"Could not find embedding column in {embeddings_file}")
        
        embeddings = np.array(df[embedding_col].tolist())
        
        # Get texts
        if "combined_text" in df.columns:
            texts = df["combined_text"].tolist()
        elif texts_df is not None and "combined_text" in texts_df.columns:
            texts = texts_df["combined_text"].tolist()
        else:
            print("Warning: No texts available, embeddings will be keyed by hash of index")
            texts = [str(i) for i in range(len(embeddings))]
            
    elif ext == ".npy":
        embeddings = np.load(embeddings_file)
        if texts_df is not None and "combined_text" in texts_df.columns:
            texts = texts_df["combined_text"].tolist()
        else:
            raise ValueError("texts_df with combined_text required for .npy files")
    else:
        raise ValueError(f"Unsupported embeddings format: {ext}")
    
    print(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Update config dimensions if needed
    if config.dimensions is None:
        config.dimensions = embeddings.shape[1]
    
    # Store in cache
    embedding_cache.store_batch(
        texts=texts,
        embeddings=embeddings,
        config=config,
    )
    
    return {
        "embeddings_imported": len(embeddings),
        "dimensions": embeddings.shape[1],
        "config": config.to_dict(),
    }


def migrate_from_working_parquet(
    working_parquet: Path,
    conference_year: int,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Migrate from a complete working parquet file that contains
    presentations, embeddings, and possibly session assignments.
    
    This is for files like 'cleaned_presentations_with_embeddings.parquet'
    from the existing workflow.
    
    Args:
        working_parquet: Path to the working parquet file
        conference_year: 2-digit year for the conference
        output_dir: Output directory for new databases (default: same as input)
        
    Returns:
        Migration statistics
    """
    if output_dir is None:
        output_dir = working_parquet.parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the parquet
    df = pd.read_parquet(working_parquet)
    print(f"Loaded working parquet: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Create new databases
    conf_name = f"AIM{conference_year}"
    cache_path = output_dir / f"{conf_name}_cache.db"
    working_path = output_dir / f"{conf_name}_working.db"
    
    cache = EmbeddingCache(cache_path)
    db = ConferenceDB(working_path, conference_year=conference_year)
    
    stats = {"source_file": str(working_parquet)}
    
    # Migrate presentations
    pres_stats = migrate_presentations(
        working_parquet, db,
        conference_year=conference_year,
    )
    stats.update(pres_stats)
    
    # Check for embeddings
    embedding_cols = [c for c in df.columns if "embed" in c.lower()]
    if embedding_cols:
        print(f"Found embedding columns: {embedding_cols}")
        
        # Try to migrate embeddings
        config = EmbeddingConfig(
            model_name="gemini-embedding-001",
            model_version="1.0",
            task_type="SEMANTIC_SIMILARITY",
        )
        
        embed_col = embedding_cols[0]
        try:
            embeddings = np.array(df[embed_col].tolist())
            
            if "combined_text" in df.columns:
                texts = df["combined_text"].tolist()
            else:
                texts = (df["Paper Title"].fillna("") + "\n\n" + 
                        df["Abstract"].fillna("")).tolist()
            
            cache.store_batch(texts=texts, embeddings=embeddings, config=config)
            stats["embeddings_imported"] = len(embeddings)
            stats["embedding_dimensions"] = embeddings.shape[1]
        except Exception as e:
            print(f"Warning: Could not migrate embeddings: {e}")
    
    stats["cache_db"] = str(cache_path)
    stats["working_db"] = str(working_path)
    
    return stats


def main():
    """Command line interface for migration."""
    parser = argparse.ArgumentParser(
        description="Migrate existing SMART data to new SQLite format"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Migration type")
    
    # Full migration from working parquet
    full_parser = subparsers.add_parser(
        "full",
        help="Full migration from a working parquet file"
    )
    full_parser.add_argument("source", type=Path, help="Source parquet file")
    full_parser.add_argument("--year", type=int, default=26, help="2-digit conference year")
    full_parser.add_argument("--output-dir", type=Path, help="Output directory")
    
    # Presentations only
    pres_parser = subparsers.add_parser(
        "presentations",
        help="Migrate presentations only"
    )
    pres_parser.add_argument("source", type=Path, help="Source file (CSV/Excel/Parquet)")
    pres_parser.add_argument("--db", type=Path, required=True, help="Target working database")
    pres_parser.add_argument("--year", type=int, default=26, help="2-digit conference year")
    
    # Embeddings only
    embed_parser = subparsers.add_parser(
        "embeddings",
        help="Migrate embeddings only"
    )
    embed_parser.add_argument("source", type=Path, help="Source embeddings file")
    embed_parser.add_argument("--cache", type=Path, required=True, help="Target cache database")
    embed_parser.add_argument("--texts", type=Path, help="File with texts for the embeddings")
    embed_parser.add_argument("--model", default="gemini-embedding-001", help="Model name")
    embed_parser.add_argument("--version", default="1.0", help="Model version")
    
    args = parser.parse_args()
    
    if args.command == "full":
        stats = migrate_from_working_parquet(
            args.source,
            conference_year=args.year,
            output_dir=args.output_dir,
        )
        print("\nMigration complete!")
        print(f"Stats: {stats}")
        
    elif args.command == "presentations":
        db = ConferenceDB(args.db, conference_year=args.year)
        stats = migrate_presentations(args.source, db, conference_year=args.year)
        print(f"\nPresentation migration complete: {stats}")
        
    elif args.command == "embeddings":
        cache = EmbeddingCache(args.cache)
        config = EmbeddingConfig(
            model_name=args.model,
            model_version=args.version,
        )
        texts_df = None
        if args.texts:
            ext = args.texts.suffix.lower()
            if ext == ".parquet":
                texts_df = pd.read_parquet(args.texts)
            elif ext == ".csv":
                texts_df = pd.read_csv(args.texts)
        
        stats = migrate_embeddings(args.source, cache, texts_df, config)
        print(f"\nEmbedding migration complete: {stats}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
Export functionality for SMART system.

Provides configurable export to various formats with field redaction
for different use cases (cloud viewer, organizer tools, spreadsheets).
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class ExportFormat(Enum):
    """Supported export formats."""
    SQLITE = "sqlite"
    PARQUET = "parquet"
    CSV = "csv"
    EXCEL = "excel"


@dataclass
class ExportProfile:
    """
    Configuration for export operations.
    
    Controls which fields are included, redacted, or encrypted.
    """
    name: str
    description: str
    
    # Fields to include (None = all non-redacted fields)
    include_fields: Optional[List[str]] = None
    
    # Fields to completely remove
    redact_fields: List[str] = field(default_factory=list)
    
    # Fields to hash/anonymize (show but not original value)
    anonymize_fields: List[str] = field(default_factory=list)
    
    # Whether to include embeddings
    include_embeddings: bool = False
    
    # Whether to include similarity matrices
    include_similarities: bool = True
    
    # Whether to include processing history
    include_history: bool = False
    
    # Output format
    format: ExportFormat = ExportFormat.PARQUET


# Predefined export profiles
PROFILE_CLOUD_VIEWER = ExportProfile(
    name="cloud_viewer",
    description="Export for public cloud viewer - no PII, no abstracts by default",
    redact_fields=[
        "presenter_email",
        "presenter_first_name",
        "presenter_last_name",
        "affiliation",
        "extra_data",
    ],
    anonymize_fields=[],
    include_embeddings=False,
    include_similarities=True,
    include_history=False,
    format=ExportFormat.PARQUET,
)

PROFILE_ORGANIZER_FULL = ExportProfile(
    name="organizer_full",
    description="Full export for organizers - includes all data",
    redact_fields=[],
    anonymize_fields=[],
    include_embeddings=True,
    include_similarities=True,
    include_history=True,
    format=ExportFormat.SQLITE,
)

PROFILE_ROOM_ASSIGNMENT = ExportProfile(
    name="room_assignment",
    description="Spreadsheet for room and time assignment",
    include_fields=[
        "session_id",
        "title",
        "keywords",
        "coherence",
        "size",
        "technical_community",
    ],
    redact_fields=[],
    include_embeddings=False,
    include_similarities=False,
    include_history=False,
    format=ExportFormat.EXCEL,
)

PROFILE_PRESENTER_LIST = ExportProfile(
    name="presenter_list",
    description="Contact list for presenter communication",
    include_fields=[
        "abstract_id",
        "title",
        "presenter_first_name",
        "presenter_last_name",
        "presenter_email",
        "session_id",
        "session_title",
    ],
    redact_fields=[],
    include_embeddings=False,
    include_similarities=False,
    format=ExportFormat.CSV,
)


def get_profile(name: str) -> ExportProfile:
    """Get a predefined export profile by name."""
    profiles = {
        "cloud_viewer": PROFILE_CLOUD_VIEWER,
        "organizer_full": PROFILE_ORGANIZER_FULL,
        "room_assignment": PROFILE_ROOM_ASSIGNMENT,
        "presenter_list": PROFILE_PRESENTER_LIST,
    }
    
    if name not in profiles:
        raise ValueError(f"Unknown profile: {name}. Available: {list(profiles.keys())}")
    
    return profiles[name]


def _apply_field_filters(
    df: pd.DataFrame,
    profile: ExportProfile,
) -> pd.DataFrame:
    """Apply field inclusion/redaction based on profile."""
    # Remove redacted fields
    for field_name in profile.redact_fields:
        if field_name in df.columns:
            df = df.drop(columns=[field_name])
    
    # Anonymize specified fields
    for field_name in profile.anonymize_fields:
        if field_name in df.columns:
            df[field_name] = df[field_name].apply(
                lambda x: f"ANON-{hash(str(x)) % 10000:04d}" if pd.notna(x) else None
            )
    
    # Keep only specified fields if include_fields is set
    if profile.include_fields:
        available = [f for f in profile.include_fields if f in df.columns]
        df = df[available]
    
    return df


def export_presentations(
    conference_db,
    profile: ExportProfile,
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Export presentations according to profile.
    
    Args:
        conference_db: ConferenceDB instance
        profile: Export profile configuration
        output_path: Path for output file
        
    Returns:
        Metadata about the export
    """
    df = conference_db.export_to_dataframe()
    df = _apply_field_filters(df, profile)
    
    output_path = Path(output_path)
    
    if profile.format == ExportFormat.PARQUET:
        df.to_parquet(output_path, compression='snappy', index=False)
    elif profile.format == ExportFormat.CSV:
        df.to_csv(output_path, index=False)
    elif profile.format == ExportFormat.EXCEL:
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format for presentations: {profile.format}")
    
    return {
        "output_path": str(output_path),
        "format": profile.format.value,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }


def export_sessions(
    conference_db,
    profile: ExportProfile,
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Export sessions according to profile.
    """
    sessions = conference_db.get_sessions()
    df = pd.DataFrame(sessions)
    df = _apply_field_filters(df, profile)
    
    output_path = Path(output_path)
    
    if profile.format == ExportFormat.PARQUET:
        df.to_parquet(output_path, compression='snappy', index=False)
    elif profile.format == ExportFormat.CSV:
        df.to_csv(output_path, index=False)
    elif profile.format == ExportFormat.EXCEL:
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format for sessions: {profile.format}")
    
    return {
        "output_path": str(output_path),
        "format": profile.format.value,
        "row_count": len(df),
        "columns": list(df.columns),
    }


def export_similarity_matrix(
    embeddings: np.ndarray,
    ids: List[str],
    output_path: str | Path,
    similarity_func=None,
) -> Dict[str, Any]:
    """
    Export pre-computed similarity matrix.
    
    Args:
        embeddings: NxD array of embeddings
        ids: List of IDs corresponding to embeddings
        output_path: Path for output file
        similarity_func: Similarity function (default: cosine_similarity)
        
    Returns:
        Metadata about the export
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    if similarity_func is None:
        similarity_func = cosine_similarity
    
    similarities = similarity_func(embeddings, embeddings)
    df = pd.DataFrame(similarities, index=ids, columns=ids)
    
    output_path = Path(output_path)
    df.to_parquet(output_path, compression='snappy')
    
    return {
        "output_path": str(output_path),
        "size": similarities.shape,
    }


def export_for_viewer(
    conference_db,
    embeddings: np.ndarray,
    abstract_ids: List[str],
    output_dir: str | Path,
    profile: Optional[ExportProfile] = None,
    version_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export complete package for cloud viewer deployment.
    
    Creates multiple files in output directory:
    - presentations.parquet
    - sessions.parquet
    - presentation_similarities.parquet (if enabled)
    - session_similarities.parquet (if enabled)
    - metadata.json
    
    Args:
        conference_db: ConferenceDB instance
        embeddings: Presentation embeddings array
        abstract_ids: List of abstract IDs
        output_dir: Output directory
        profile: Export profile (default: cloud_viewer)
        version_tag: Optional version string for metadata
        
    Returns:
        Metadata about all exported files
    """
    from datetime import datetime
    from sklearn.metrics.pairwise import cosine_similarity
    
    profile = profile or PROFILE_CLOUD_VIEWER
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "profile": profile.name,
        "version": version_tag or datetime.now().strftime("%Y%m%d_%H%M%S"),
        "exported_at": datetime.now().isoformat(),
        "files": {},
    }
    
    # Export presentations
    pres_path = output_dir / "presentations.parquet"
    pres_result = export_presentations(conference_db, profile, pres_path)
    results["files"]["presentations"] = pres_result
    
    # Export sessions
    sessions_path = output_dir / "sessions.parquet"
    sessions_result = export_sessions(conference_db, profile, sessions_path)
    results["files"]["sessions"] = sessions_result
    
    # Export similarity matrices if enabled
    if profile.include_similarities:
        # Presentation similarities
        pres_sim_path = output_dir / "presentation_similarities.parquet"
        pres_sim_result = export_similarity_matrix(
            embeddings, abstract_ids, pres_sim_path
        )
        results["files"]["presentation_similarities"] = pres_sim_result
        
        # Session similarities
        sessions = conference_db.get_sessions()
        if sessions:
            id_to_idx = {aid: i for i, aid in enumerate(abstract_ids)}
            session_embeddings = []
            session_ids = []
            
            for session in sessions:
                # Get presentations for this session
                pres = conference_db.get_presentations(session_id=session["session_id"])
                if pres:
                    indices = [id_to_idx[p["abstract_id"]] for p in pres 
                              if p["abstract_id"] in id_to_idx]
                    if indices:
                        session_emb = np.mean(embeddings[indices], axis=0)
                        session_embeddings.append(session_emb)
                        session_ids.append(session["session_id"])
            
            if session_embeddings:
                session_emb_array = np.array(session_embeddings)
                sess_sim_path = output_dir / "session_similarities.parquet"
                sess_sim_result = export_similarity_matrix(
                    session_emb_array, session_ids, sess_sim_path
                )
                results["files"]["session_similarities"] = sess_sim_result
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(results, f, indent=2)
    results["files"]["metadata"] = {"output_path": str(metadata_path)}
    
    return results


def export_for_organizers(
    conference_db,
    embedding_cache,
    output_path: str | Path,
    include_cache: bool = True,
) -> Dict[str, Any]:
    """
    Export complete SQLite database for organizers.
    
    Creates a single SQLite file containing all data for backup
    and transfer purposes.
    
    Args:
        conference_db: ConferenceDB instance
        embedding_cache: EmbeddingCache instance
        output_path: Path for output SQLite file
        include_cache: Whether to include embedding cache
        
    Returns:
        Metadata about the export
    """
    import shutil
    from datetime import datetime
    
    output_path = Path(output_path)
    
    # Copy the working database
    shutil.copy2(conference_db.db_path, output_path)
    
    results = {
        "output_path": str(output_path),
        "exported_at": datetime.now().isoformat(),
        "stats": conference_db.get_stats(),
    }
    
    # Optionally attach and copy cache data
    if include_cache and embedding_cache:
        with sqlite3.connect(output_path) as conn:
            # Attach cache database
            conn.execute(
                f"ATTACH DATABASE '{embedding_cache.db_path}' AS cache_db"
            )
            
            # Create cache tables in export
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exported_embeddings AS
                SELECT * FROM cache_db.embeddings
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exported_model_registry AS
                SELECT * FROM cache_db.model_registry
            """)
            
            conn.execute("DETACH DATABASE cache_db")
            conn.commit()
        
        results["cache_stats"] = embedding_cache.get_stats()
    
    return results


def export_spreadsheet(
    conference_db,
    output_path: str | Path,
    include_sessions: bool = True,
    include_presentations: bool = True,
    profile: Optional[ExportProfile] = None,
) -> Dict[str, Any]:
    """
    Export to Excel spreadsheet with multiple sheets.
    
    Creates an Excel file with separate sheets for:
    - Sessions (if enabled)
    - Presentations (if enabled)
    
    Args:
        conference_db: ConferenceDB instance
        output_path: Path for Excel file
        include_sessions: Include sessions sheet
        include_presentations: Include presentations sheet
        profile: Optional export profile for field filtering
        
    Returns:
        Metadata about the export
    """
    profile = profile or PROFILE_ROOM_ASSIGNMENT
    output_path = Path(output_path)
    
    results = {
        "output_path": str(output_path),
        "sheets": {},
    }
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        if include_sessions:
            sessions = conference_db.get_sessions()
            df_sessions = pd.DataFrame(sessions)
            df_sessions = _apply_field_filters(df_sessions, profile)
            df_sessions.to_excel(writer, sheet_name='Sessions', index=False)
            results["sheets"]["Sessions"] = {
                "row_count": len(df_sessions),
                "columns": list(df_sessions.columns),
            }
        
        if include_presentations:
            df_pres = conference_db.export_to_dataframe()
            df_pres = _apply_field_filters(df_pres, profile)
            df_pres.to_excel(writer, sheet_name='Presentations', index=False)
            results["sheets"]["Presentations"] = {
                "row_count": len(df_pres),
                "columns": list(df_pres.columns),
            }
    
    return results


# ============================================================
# Viewer Bundle Export/Import
# ============================================================

@dataclass
class ViewerBundle:
    """
    Container for viewer data bundle.
    
    A ViewerBundle contains all data needed by the session viewer app
    in a single, validated package.
    """
    # Core data
    presentations: pd.DataFrame
    sessions: pd.DataFrame
    pres_similarities: pd.DataFrame
    session_similarities: pd.DataFrame
    
    # Optional data
    embeddings: Optional[np.ndarray] = None
    abstract_ids: Optional[List[str]] = None
    
    # Metadata
    manifest: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """
        Validate bundle integrity.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check presentations DataFrame
        required_pres_cols = ["Abstract ID", "Title", "Session Code"]
        for col in required_pres_cols:
            if col not in self.presentations.columns:
                errors.append(f"Missing required column in presentations: {col}")
        
        # Check sessions DataFrame
        required_sess_cols = ["session_id", "session_size", "session_coherence"]
        # Also accept 'cluster_id' for backwards compatibility
        if "session_id" not in self.sessions.columns and "cluster_id" not in self.sessions.columns:
            errors.append("Missing session identifier column (session_id or cluster_id)")
        for col in ["session_size", "session_coherence"]:
            if col not in self.sessions.columns:
                errors.append(f"Missing required column in sessions: {col}")
        
        # Check similarity matrix dimensions
        n_pres = len(self.presentations)
        if self.pres_similarities.shape != (n_pres, n_pres):
            errors.append(
                f"Presentation similarities shape {self.pres_similarities.shape} "
                f"doesn't match presentation count {n_pres}"
            )
        
        n_sess = len(self.sessions)
        if self.session_similarities.shape != (n_sess, n_sess):
            errors.append(
                f"Session similarities shape {self.session_similarities.shape} "
                f"doesn't match session count {n_sess}"
            )
        
        # Check embeddings if present
        if self.embeddings is not None:
            if len(self.embeddings) != n_pres:
                errors.append(
                    f"Embeddings count {len(self.embeddings)} "
                    f"doesn't match presentation count {n_pres}"
                )
            if self.abstract_ids is None:
                errors.append("Embeddings present but abstract_ids missing")
            elif len(self.abstract_ids) != len(self.embeddings):
                errors.append("abstract_ids length doesn't match embeddings")
        
        return errors


def export_viewer_bundle(
    df_presentations: pd.DataFrame,
    df_sessions: pd.DataFrame,
    embeddings: np.ndarray,
    abstract_ids: List[str],
    output_path: str | Path,
    conference_name: str = "conference",
    version_tag: Optional[str] = None,
    include_abstracts: bool = False,
    include_embeddings: bool = True,
    encrypt_password: Optional[str] = None,
    recalculate_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Export a complete viewer bundle as a directory or zip file.
    
    Creates a bundle containing:
    - manifest.json (metadata and validation info)
    - presentations.parquet
    - sessions.parquet
    - pres_similarities.parquet
    - session_similarities.parquet
    - embeddings.npz (if include_embeddings=True)
    - presentations_encrypted.crypt (if encrypt_password provided)
    
    Args:
        df_presentations: Presentations DataFrame with session assignments
        df_sessions: Sessions DataFrame with metrics
        embeddings: NxD array of presentation embeddings
        abstract_ids: List of abstract IDs corresponding to embeddings
        output_path: Output directory or .zip file path
        conference_name: Name for the conference (used in manifest)
        version_tag: Optional version string
        include_abstracts: Whether to include abstracts in public export
        include_embeddings: Whether to include embeddings for future calculations
        encrypt_password: Password to encrypt sensitive data (abstracts, PII)
        recalculate_metrics: Whether to recalculate presentation fit metrics.
            Set to False when exporting existing data for viewing only.
        
    Returns:
        Dict with export metadata
    """
    from datetime import datetime
    from sklearn.metrics.pairwise import cosine_similarity
    import hashlib
    import gzip
    
    output_path = Path(output_path)
    is_zip = output_path.suffix.lower() == '.zip'
    
    # Create output directory
    if is_zip:
        bundle_dir = output_path.with_suffix('')
    else:
        bundle_dir = output_path
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare manifest
    manifest = {
        "bundle_version": "1.0",
        "conference_name": conference_name,
        "version_tag": version_tag or datetime.now().strftime("%Y%m%d_%H%M%S"),
        "created_at": datetime.now().isoformat(),
        "presentation_count": len(df_presentations),
        "session_count": len(df_sessions),
        "has_embeddings": include_embeddings,
        "has_encrypted_data": encrypt_password is not None,
        "files": {},
    }
    
    # ---- Export presentations ----
    # Rename columns for viewer compatibility
    df_pres_export = df_presentations.copy()
    
    # Standard column renames for viewer
    column_renames = {
        "abstract_id": "Abstract ID",
        "title": "Title",
        "session_id": "Session Code",
        "presentation_fit": "Presentation Session Fit",
        "fit_raw_deviation": "Presentation Raw Deviation",
        "fit_std_deviation": "Presentation Standardized Deviation",
        "session_std_dev": "Session Std Dev",
    }
    
    for old_name, new_name in column_renames.items():
        if old_name in df_pres_export.columns and new_name not in df_pres_export.columns:
            df_pres_export = df_pres_export.rename(columns={old_name: new_name})
    
    # Remove any "Unnamed" columns (artifacts from CSV import)
    unnamed_cols = [col for col in df_pres_export.columns if col.startswith('Unnamed')]
    if unnamed_cols:
        df_pres_export = df_pres_export.drop(columns=unnamed_cols)
    
    # Metric columns that may need to be calculated
    metric_columns = [
        "Presentation Session Fit",
        "Presentation Raw Deviation",
        "Presentation Standardized Deviation",
        "Session Std Dev",
    ]
    
    # Only handle metric recalculation if requested
    if recalculate_metrics:
        # If input already has these columns, rename them to preserve original values
        for col in metric_columns:
            if col in df_pres_export.columns:
                df_pres_export = df_pres_export.rename(columns={col: f"{col} (input)"})
        
        # Calculate new metrics
        df_pres_export = _calculate_presentation_metrics(
            df_pres_export, embeddings, abstract_ids
        )
    # If not recalculating, just ensure the expected column names exist
    # (metrics should already be present from input)
    
    # Ensure Abstract ID is properly typed for similarity lookups
    if "Abstract ID" in df_pres_export.columns:
        # Handle NA values - use nullable Int64 dtype or keep as-is if NAs present
        if df_pres_export["Abstract ID"].isna().any():
            # Use nullable integer type to preserve NAs
            df_pres_export["Abstract ID"] = pd.to_numeric(
                df_pres_export["Abstract ID"], errors='coerce'
            ).astype("Int64")
        else:
            try:
                df_pres_export["Abstract ID"] = df_pres_export["Abstract ID"].astype(int)
            except (ValueError, TypeError):
                pass  # Keep as-is if conversion fails
    
    # Create public version (no PII, optionally no abstracts)
    pii_columns = ["presenter_email", "presenter_first_name", "presenter_last_name", 
                   "affiliation", "Owner-E-mail Address", "Owner-First Name", "Owner-Last Name"]
    
    df_public = df_pres_export.drop(
        columns=[c for c in pii_columns if c in df_pres_export.columns],
        errors='ignore'
    )
    
    if not include_abstracts:
        abstract_cols = ["abstract", "Abstract"]
        df_public = df_public.drop(
            columns=[c for c in abstract_cols if c in df_public.columns],
            errors='ignore'
        )
    
    # Save public presentations
    pres_path = bundle_dir / "presentations.parquet"
    df_public.to_parquet(pres_path, compression='snappy', index=False)
    manifest["files"]["presentations"] = {
        "filename": "presentations.parquet",
        "rows": len(df_public),
        "columns": list(df_public.columns),
        "checksum": _file_checksum(pres_path),
    }
    
    # Save encrypted version if password provided
    if encrypt_password:
        try:
            import cryptpandas as crp
            encrypted_path = bundle_dir / "presentations_encrypted.crypt"
            crp.write_encrypted(df_pres_export, path=str(encrypted_path), password=encrypt_password)
            manifest["files"]["presentations_encrypted"] = {
                "filename": "presentations_encrypted.crypt",
                "rows": len(df_pres_export),
            }
        except ImportError:
            pass  # cryptpandas not available
    
    # ---- Export sessions ----
    df_sess_export = df_sessions.copy()
    
    # Standard column renames for viewer
    sess_renames = {
        "session_id": "cluster_id",  # Viewer expects cluster_id
    }
    for old_name, new_name in sess_renames.items():
        if old_name in df_sess_export.columns and new_name not in df_sess_export.columns:
            df_sess_export = df_sess_export.rename(columns={old_name: new_name})
    
    # Ensure session_id or cluster_id exists
    if "cluster_id" not in df_sess_export.columns and "session_id" in df_sess_export.columns:
        df_sess_export["cluster_id"] = df_sess_export["session_id"]
    
    sess_path = bundle_dir / "sessions.parquet"
    df_sess_export.to_parquet(sess_path, compression='snappy', index=False)
    manifest["files"]["sessions"] = {
        "filename": "sessions.parquet",
        "rows": len(df_sess_export),
        "columns": list(df_sess_export.columns),
        "checksum": _file_checksum(sess_path),
    }
    
    # ---- Export similarity matrices ----
    # Clean and deduplicate abstract_ids for similarity matrix
    # Handle NaN, duplicates, and convert floats to ints
    clean_ids = []
    clean_indices = []
    seen_ids = set()
    
    for i, aid in enumerate(abstract_ids):
        # Skip NaN/None values
        if aid is None or (isinstance(aid, float) and np.isnan(aid)) or str(aid).lower() == 'nan':
            continue
        
        # Convert to clean ID (int if possible, else string)
        try:
            clean_id = int(float(aid))
        except (ValueError, TypeError):
            clean_id = str(aid)
        
        # Skip duplicates
        if clean_id in seen_ids:
            continue
        
        seen_ids.add(clean_id)
        clean_ids.append(clean_id)
        clean_indices.append(i)
    
    # Create abstract_id to index mapping using clean indices
    id_to_idx = {str(clean_ids[i]): clean_indices[i] for i in range(len(clean_ids))}
    
    # Presentation similarities - use only clean (non-duplicate, non-NaN) entries
    clean_embeddings = embeddings[clean_indices]
    pres_sim = cosine_similarity(clean_embeddings, clean_embeddings)
    
    df_pres_sim = pd.DataFrame(pres_sim, index=clean_ids, columns=clean_ids)
    pres_sim_path = bundle_dir / "pres_similarities.parquet"
    df_pres_sim.to_parquet(pres_sim_path, compression='snappy')
    manifest["files"]["pres_similarities"] = {
        "filename": "pres_similarities.parquet",
        "shape": list(df_pres_sim.shape),
        "checksum": _file_checksum(pres_sim_path),
    }
    
    # Session similarities
    session_id_col = "cluster_id" if "cluster_id" in df_sess_export.columns else "session_id"
    session_ids = df_sess_export[session_id_col].tolist()
    session_col = "Session Code" if "Session Code" in df_pres_export.columns else "session_id"
    
    session_centroids = []
    valid_session_ids = []
    
    for sess_id in session_ids:
        # Find presentations in this session
        if session_col in df_pres_export.columns:
            sess_mask = df_pres_export[session_col] == sess_id
            sess_abstract_ids = df_pres_export.loc[sess_mask, "Abstract ID"].tolist() if "Abstract ID" in df_pres_export.columns else []
        else:
            sess_abstract_ids = []
        
        # Get embeddings for these presentations
        indices = []
        for aid in sess_abstract_ids:
            aid_str = str(aid)
            if aid_str in id_to_idx:
                indices.append(id_to_idx[aid_str])
        
        if indices:
            centroid = np.mean(embeddings[indices], axis=0)
            session_centroids.append(centroid)
            valid_session_ids.append(sess_id)
    
    if session_centroids:
        session_centroids_array = np.array(session_centroids)
        sess_sim = cosine_similarity(session_centroids_array, session_centroids_array)
        df_sess_sim = pd.DataFrame(sess_sim, index=valid_session_ids, columns=valid_session_ids)
    else:
        df_sess_sim = pd.DataFrame()
    
    sess_sim_path = bundle_dir / "session_similarities.parquet"
    df_sess_sim.to_parquet(sess_sim_path, compression='snappy')
    manifest["files"]["session_similarities"] = {
        "filename": "session_similarities.parquet",
        "shape": list(df_sess_sim.shape),
        "checksum": _file_checksum(sess_sim_path),
    }
    
    # ---- Export embeddings ----
    if include_embeddings:
        emb_path = bundle_dir / "embeddings.npz"
        np.savez_compressed(
            emb_path,
            embeddings=embeddings,
            abstract_ids=np.array(abstract_ids, dtype=object),
        )
        manifest["files"]["embeddings"] = {
            "filename": "embeddings.npz",
            "shape": list(embeddings.shape),
            "checksum": _file_checksum(emb_path),
        }
    
    # ---- Save manifest ----
    manifest_path = bundle_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # ---- Create zip if requested ----
    if is_zip:
        import shutil
        shutil.make_archive(str(output_path.with_suffix('')), 'zip', bundle_dir)
        shutil.rmtree(bundle_dir)
        manifest["bundle_path"] = str(output_path)
    else:
        manifest["bundle_path"] = str(bundle_dir)
    
    return manifest


def load_viewer_bundle(bundle_path: str | Path) -> ViewerBundle:
    """
    Load a viewer bundle from directory or zip file.
    
    Args:
        bundle_path: Path to bundle directory or .zip file
        
    Returns:
        ViewerBundle with all loaded data
    """
    import tempfile
    import shutil
    
    bundle_path = Path(bundle_path)
    
    # Handle zip files
    if bundle_path.suffix.lower() == '.zip':
        temp_dir = tempfile.mkdtemp()
        shutil.unpack_archive(bundle_path, temp_dir)
        # Find the actual bundle directory (may be nested)
        contents = list(Path(temp_dir).iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            bundle_dir = contents[0]
        else:
            bundle_dir = Path(temp_dir)
        cleanup_temp = True
    else:
        bundle_dir = bundle_path
        cleanup_temp = False
    
    try:
        # Load manifest
        manifest_path = bundle_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {}
        
        # Load presentations
        pres_path = bundle_dir / "presentations.parquet"
        df_presentations = pd.read_parquet(pres_path)
        
        # Load sessions
        sess_path = bundle_dir / "sessions.parquet"
        df_sessions = pd.read_parquet(sess_path)
        
        # Load similarities
        pres_sim_path = bundle_dir / "pres_similarities.parquet"
        df_pres_sim = pd.read_parquet(pres_sim_path)
        
        sess_sim_path = bundle_dir / "session_similarities.parquet"
        df_sess_sim = pd.read_parquet(sess_sim_path)
        
        # Load embeddings if present
        embeddings = None
        abstract_ids = None
        emb_path = bundle_dir / "embeddings.npz"
        if emb_path.exists():
            data = np.load(emb_path, allow_pickle=True)
            embeddings = data["embeddings"]
            abstract_ids = list(data["abstract_ids"])
        
        return ViewerBundle(
            presentations=df_presentations,
            sessions=df_sessions,
            pres_similarities=df_pres_sim,
            session_similarities=df_sess_sim,
            embeddings=embeddings,
            abstract_ids=abstract_ids,
            manifest=manifest,
        )
    
    finally:
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _file_checksum(path: Path) -> str:
    """Calculate MD5 checksum of a file."""
    import hashlib
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


def _calculate_presentation_metrics(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    abstract_ids: List[str],
) -> pd.DataFrame:
    """
    Calculate presentation fit metrics and add to DataFrame.
    
    Adds columns:
    - Presentation Session Fit
    - Presentation Raw Deviation
    - Presentation Standardized Deviation
    - Session Std Dev
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    df = df.copy()
    
    # Build ID to embedding index map
    id_to_idx = {}
    for i, aid in enumerate(abstract_ids):
        id_to_idx[str(aid)] = i
        try:
            id_to_idx[int(aid)] = i
        except (ValueError, TypeError):
            pass
    
    # Determine session column
    session_col = None
    for col in ["Session Code", "session_id", "cluster_id"]:
        if col in df.columns:
            session_col = col
            break
    
    if session_col is None:
        return df
    
    # Determine abstract ID column
    id_col = None
    for col in ["Abstract ID", "abstract_id"]:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        return df
    
    # Calculate similarity matrix
    sim_matrix = cosine_similarity(embeddings, embeddings)
    
    # Group by session
    session_groups = df.groupby(session_col)
    
    fit_scores = []
    raw_devs = []
    std_devs = []
    session_std_devs = []
    
    for _, row in df.iterrows():
        abstract_id = row[id_col]
        session_id = row[session_col]
        
        # Get presentation index
        pres_idx = id_to_idx.get(abstract_id) or id_to_idx.get(str(abstract_id))
        
        if pres_idx is None or pd.isna(session_id):
            fit_scores.append(np.nan)
            raw_devs.append(np.nan)
            std_devs.append(np.nan)
            session_std_devs.append(np.nan)
            continue
        
        # Get session presentation indices
        session_mask = df[session_col] == session_id
        session_ids = df.loc[session_mask, id_col].tolist()
        session_indices = [
            id_to_idx.get(aid) or id_to_idx.get(str(aid))
            for aid in session_ids
        ]
        session_indices = [i for i in session_indices if i is not None]
        
        if len(session_indices) < 2:
            fit_scores.append(1.0)
            raw_devs.append(0.0)
            std_devs.append(0.0)
            session_std_devs.append(0.0)
            continue
        
        # Calculate fit (mean similarity to other session members)
        other_indices = [i for i in session_indices if i != pres_idx]
        if other_indices:
            fit = np.mean(sim_matrix[pres_idx, other_indices])
        else:
            fit = 1.0
        
        # Calculate session coherence
        n = len(session_indices)
        coherence_sum = 0.0
        count = 0
        for i, idx_i in enumerate(session_indices):
            for j, idx_j in enumerate(session_indices):
                if i < j:
                    coherence_sum += sim_matrix[idx_i, idx_j]
                    count += 1
        coherence = coherence_sum / count if count > 0 else 1.0
        
        # Calculate session std dev of fit scores
        all_fits = []
        for idx in session_indices:
            other = [i for i in session_indices if i != idx]
            if other:
                all_fits.append(np.mean(sim_matrix[idx, other]))
        session_std = np.std(all_fits, ddof=1) if len(all_fits) > 1 else 0.0
        
        raw_dev = fit - coherence
        std_dev = raw_dev / session_std if session_std > 0 else 0.0
        
        fit_scores.append(float(fit))
        raw_devs.append(float(raw_dev))
        std_devs.append(float(std_dev))
        session_std_devs.append(float(session_std))
    
    df["Presentation Session Fit"] = fit_scores
    df["Presentation Raw Deviation"] = raw_devs
    df["Presentation Standardized Deviation"] = std_devs
    df["Session Std Dev"] = session_std_devs
    
    return df

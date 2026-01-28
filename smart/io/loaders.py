"""
Data loaders for SMART system.

Handles loading presentation and committee data from various file formats
with flexible column mapping and automatic ID generation.
"""

import re
import hashlib
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class ColumnMapping:
    """
    Mapping configuration for loading data files.
    
    Maps source column names to standardized internal names.
    """
    # Required mappings
    title: str
    abstract: Optional[str] = None
    submission_id: Optional[str] = None
    
    # Optional presenter info
    presenter_first_name: Optional[str] = None
    presenter_last_name: Optional[str] = None
    presenter_email: Optional[str] = None
    affiliation: Optional[str] = None
    
    # Optional metadata
    technical_community: Optional[str] = None
    session_preference: Optional[str] = None
    session: Optional[str] = None  # For hybrid sessions
    
    # Extra columns to preserve (source_name -> target_name)
    extra_columns: Dict[str, str] = field(default_factory=dict)
    
    def to_rename_dict(self) -> Dict[str, str]:
        """Generate pandas rename dictionary."""
        mapping = {}
        
        if self.title:
            mapping[self.title] = "title"
        if self.abstract:
            mapping[self.abstract] = "abstract"
        if self.submission_id:
            mapping[self.submission_id] = "submission_id"
        if self.presenter_first_name:
            mapping[self.presenter_first_name] = "presenter_first_name"
        if self.presenter_last_name:
            mapping[self.presenter_last_name] = "presenter_last_name"
        if self.presenter_email:
            mapping[self.presenter_email] = "presenter_email"
        if self.affiliation:
            mapping[self.affiliation] = "affiliation"
        if self.technical_community:
            mapping[self.technical_community] = "technical_community"
        if self.session_preference:
            mapping[self.session_preference] = "session_preference"
        if self.session:
            mapping[self.session] = "session"
        
        mapping.update(self.extra_columns)
        
        return mapping


class ColumnMapper:
    """
    Interactive column mapper for data files.
    
    Helps identify and map columns from source files to standard schema.
    Can auto-detect common column patterns.
    """
    
    # Common column name patterns
    PATTERNS = {
        "title": [
            r"title",
            r"presentation.?title",
            r"paper.?title",
            r"abstract.?title",
        ],
        "abstract": [
            r"abstract",
            r"summary",
            r"description",
        ],
        "submission_id": [
            r"submission.?id",
            r"abstract.?id",
            r"paper.?id",
            r"id",
        ],
        "presenter_first_name": [
            r"first.?name",
            r"presenter.?first",
            r"author.?first",
            r"owner.?first",
        ],
        "presenter_last_name": [
            r"last.?name",
            r"presenter.?last",
            r"author.?last",
            r"owner.?last",
        ],
        "presenter_email": [
            r"e?.?mail",
            r"presenter.?email",
            r"author.?email",
            r"owner.?email",
        ],
        "affiliation": [
            r"affiliation",
            r"institution",
            r"university",
            r"company",
            r"organization",
        ],
        "technical_community": [
            r"technical.?community",
            r"community",
            r"track",
            r"category",
        ],
        "session_preference": [
            r"session.?preference",
            r"presentation.?type",
            r"oral.?poster",
        ],
        "session": [
            r"^session$",
            r"session.?name",
            r"assigned.?session",
        ],
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize mapper with source dataframe.
        
        Args:
            df: Source dataframe to map columns from
        """
        self.df = df
        self.columns = list(df.columns)
        self._auto_mapping = self._auto_detect()
    
    def _auto_detect(self) -> Dict[str, str]:
        """Auto-detect column mappings based on patterns."""
        mapping = {}
        used_columns = set()
        
        for field_name, patterns in self.PATTERNS.items():
            for col in self.columns:
                if col in used_columns:
                    continue
                col_lower = col.lower()
                for pattern in patterns:
                    if re.search(pattern, col_lower):
                        mapping[field_name] = col
                        used_columns.add(col)
                        break
                if field_name in mapping:
                    break
        
        return mapping
    
    def get_suggestions(self) -> Dict[str, Optional[str]]:
        """Get auto-detected mapping suggestions."""
        return self._auto_mapping.copy()
    
    def get_unmapped_columns(self) -> List[str]:
        """Get columns not yet mapped to standard fields."""
        mapped = set(self._auto_mapping.values())
        return [c for c in self.columns if c not in mapped]
    
    def preview_column(self, column: str, n: int = 5) -> List[Any]:
        """Preview values from a column."""
        if column not in self.columns:
            raise ValueError(f"Column '{column}' not found")
        return self.df[column].head(n).tolist()
    
    def create_mapping(
        self,
        title: Optional[str] = None,
        abstract: Optional[str] = None,
        submission_id: Optional[str] = None,
        **kwargs
    ) -> ColumnMapping:
        """
        Create column mapping, using auto-detected values as defaults.
        
        Args:
            title: Override for title column
            abstract: Override for abstract column
            submission_id: Override for submission ID column
            **kwargs: Additional field mappings
            
        Returns:
            ColumnMapping instance
        """
        # Start with auto-detected
        mapping_dict = self._auto_mapping.copy()
        
        # Apply overrides
        if title is not None:
            mapping_dict["title"] = title
        if abstract is not None:
            mapping_dict["abstract"] = abstract
        if submission_id is not None:
            mapping_dict["submission_id"] = submission_id
        
        for key, value in kwargs.items():
            if value is not None:
                mapping_dict[key] = value
        
        # Validate title is present
        if "title" not in mapping_dict or mapping_dict["title"] is None:
            raise ValueError("Title column mapping is required")
        
        return ColumnMapping(
            title=mapping_dict.get("title"),
            abstract=mapping_dict.get("abstract"),
            submission_id=mapping_dict.get("submission_id"),
            presenter_first_name=mapping_dict.get("presenter_first_name"),
            presenter_last_name=mapping_dict.get("presenter_last_name"),
            presenter_email=mapping_dict.get("presenter_email"),
            affiliation=mapping_dict.get("affiliation"),
            technical_community=mapping_dict.get("technical_community"),
            session_preference=mapping_dict.get("session_preference"),
            session=mapping_dict.get("session"),
        )


def read_file(file_path: str | Path) -> pd.DataFrame:
    """
    Read CSV or Excel file into DataFrame.
    
    Args:
        file_path: Path to file
        
    Returns:
        pandas DataFrame
    """
    path = Path(file_path)
    
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    elif path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            "Use CSV (.csv) or Excel (.xlsx, .xls)"
        )


def generate_abstract_id(
    submission_id: Optional[str],
    conference_year: int,
    temp_counter: int
) -> Tuple[str, bool]:
    """
    Generate abstract ID from submission ID or create temporary ID.
    
    Args:
        submission_id: Original submission portal ID
        conference_year: Two-digit year
        temp_counter: Counter for temporary IDs
        
    Returns:
        Tuple of (abstract_id, is_temp)
    """
    year_prefix = str(conference_year).zfill(2)
    
    if submission_id is not None and pd.notna(submission_id):
        # Try to extract numeric portion
        submission_str = str(submission_id)
        numeric_part = ''.join(filter(str.isdigit, submission_str))
        
        if numeric_part:
            # Use last 5 digits if longer
            numeric_part = numeric_part[-5:].zfill(5)
            return f"{year_prefix}{numeric_part}", False
    
    # Generate temp ID
    return f"TEMP-{year_prefix}-{str(temp_counter).zfill(5)}", True


def load_presentations(
    file_path: str | Path,
    mapping: ColumnMapping,
    conference_year: int = 26,
    drop_missing_title: bool = True,
    drop_missing_abstract: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load presentations from file with column mapping.
    
    Args:
        file_path: Path to CSV or Excel file
        mapping: Column mapping configuration
        conference_year: Two-digit year for ID generation
        drop_missing_title: Drop rows without titles
        drop_missing_abstract: Drop rows without abstracts
        
    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    # Read file
    df = read_file(file_path)
    original_count = len(df)
    
    # Apply column renaming
    rename_dict = mapping.to_rename_dict()
    df = df.rename(columns=rename_dict)
    
    # Drop rows with missing required fields
    if drop_missing_title and "title" in df.columns:
        df = df.dropna(subset=["title"])
    
    if drop_missing_abstract and "abstract" in df.columns:
        df = df.dropna(subset=["abstract"])
    
    # Generate abstract IDs
    temp_counter = 0
    abstract_ids = []
    is_temp_flags = []
    
    for _, row in df.iterrows():
        submission_id = row.get("submission_id")
        abstract_id, is_temp = generate_abstract_id(
            submission_id, conference_year, temp_counter
        )
        if is_temp:
            temp_counter += 1
        abstract_ids.append(abstract_id)
        is_temp_flags.append(is_temp)
    
    df["abstract_id"] = abstract_ids
    df["is_temp_id"] = is_temp_flags
    
    # Create combined text for embedding
    if "abstract" in df.columns:
        df["combined_text"] = df.apply(
            lambda r: f"{r['title']}: {r['abstract']}" 
            if pd.notna(r.get('abstract')) else r['title'],
            axis=1
        )
    else:
        df["combined_text"] = df["title"]
    
    metadata = {
        "source_file": str(file_path),
        "original_count": original_count,
        "loaded_count": len(df),
        "temp_id_count": sum(is_temp_flags),
        "real_id_count": len(df) - sum(is_temp_flags),
        "columns_mapped": list(rename_dict.keys()),
    }
    
    return df, metadata


def load_hybrid_sessions(
    file_path: str | Path,
    mapping: ColumnMapping,
    conference_year: int = 26,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load hybrid sessions with pre-assigned presentations.
    
    Handles presentations that span multiple time slots by assigning unique
    slot-based IDs. If a presentation appears multiple times (e.g., a 2-hour
    talk in 1-hour slots), each occurrence gets a unique ID like:
    - First occurrence: 2600001-INV
    - Second occurrence: 2600001-INV-S2
    - Third occurrence: 2600001-INV-S3
    
    All hybrid/invited presentations are marked with "-INV" suffix to distinguish
    them from regular submissions.
    
    The original abstract_id is preserved in 'source_abstract_id' field.
    
    Args:
        file_path: Path to file with hybrid session data
        mapping: Column mapping (must include 'session' field)
        conference_year: Two-digit year for ID generation
        
    Returns:
        Tuple of (presentations_df, sessions_df, metadata)
    """
    if mapping.session is None:
        raise ValueError("Session column mapping required for hybrid sessions")
    
    # Load presentations (this generates initial abstract_ids)
    df, metadata = load_presentations(file_path, mapping, conference_year)
    
    # Handle duplicate abstract_ids (presentations spanning multiple slots)
    # Track occurrence count for each abstract_id
    id_occurrence_count = {}
    slot_ids = []
    source_ids = []
    slot_numbers = []
    
    for idx, row in df.iterrows():
        original_id = row["abstract_id"]
        source_ids.append(original_id)
        
        # Count occurrences
        if original_id not in id_occurrence_count:
            id_occurrence_count[original_id] = 0
        id_occurrence_count[original_id] += 1
        occurrence = id_occurrence_count[original_id]
        
        # All invited presentations get -INV suffix
        # First occurrence: ID-INV, subsequent: ID-INV-S2, ID-INV-S3, etc.
        if occurrence == 1:
            slot_ids.append(f"{original_id}-INV")
            slot_numbers.append(1)
        else:
            slot_ids.append(f"{original_id}-INV-S{occurrence}")
            slot_numbers.append(occurrence)
    
    # Update dataframe with slot-aware IDs and invited marker
    df["source_abstract_id"] = source_ids  # Original ID for reference
    df["abstract_id"] = slot_ids  # Unique ID including -INV and slot suffix
    df["slot_number"] = slot_numbers  # Which slot this is (1, 2, 3, ...)
    df["is_invited"] = True  # Mark all as invited presentations
    
    # Count multi-slot presentations
    multi_slot_presentations = sum(1 for count in id_occurrence_count.values() if count > 1)
    total_extra_slots = sum(count - 1 for count in id_occurrence_count.values() if count > 1)
    
    # Extract unique sessions
    sessions_data = []
    for session_name in df["session"].unique():
        if pd.notna(session_name):
            session_pres = df[df["session"] == session_name]
            sessions_data.append({
                "session_id": f"HYBRID-{len(sessions_data)+1:03d}",
                "title": session_name,
                "is_hybrid": True,
                "presentation_count": len(session_pres),
                "presentation_ids": session_pres["abstract_id"].tolist(),
                # Track unique presentations vs total slots
                "unique_presentation_count": session_pres["source_abstract_id"].nunique(),
                "total_slots": len(session_pres),
            })
    
    df_sessions = pd.DataFrame(sessions_data)
    
    # Add session_id to presentations
    session_id_map = {
        row["title"]: row["session_id"] 
        for _, row in df_sessions.iterrows()
    }
    df["session_id"] = df["session"].map(session_id_map)
    
    metadata["session_count"] = len(df_sessions)
    metadata["is_hybrid"] = True
    metadata["multi_slot_presentations"] = multi_slot_presentations
    metadata["total_extra_slots"] = total_extra_slots
    metadata["unique_presentations"] = len(id_occurrence_count)
    metadata["total_presentation_slots"] = len(df)
    
    return df, df_sessions, metadata


def load_committees(
    file_path: str | Path,
    name_column: str = "Committee_Name",
    description_column: str = "Description",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load committee data for session assignment.
    
    Args:
        file_path: Path to committee data file
        name_column: Column containing committee names
        description_column: Column containing descriptions
        
    Returns:
        Tuple of (DataFrame, metadata)
    """
    df = read_file(file_path)
    original_count = len(df)
    
    # Validate columns
    if name_column not in df.columns:
        raise ValueError(f"Column '{name_column}' not found in file")
    if description_column not in df.columns:
        raise ValueError(f"Column '{description_column}' not found in file")
    
    # Rename and select columns
    df = df.rename(columns={
        name_column: "committee_name",
        description_column: "description",
    })
    
    # Drop rows with missing data
    df = df.dropna(subset=["committee_name", "description"])
    
    # Generate committee IDs
    df["committee_id"] = [f"COM-{i+1:03d}" for i in range(len(df))]
    
    # Create combined text for embedding
    df["combined_text"] = df["committee_name"] + ": " + df["description"]
    
    metadata = {
        "source_file": str(file_path),
        "original_count": original_count,
        "loaded_count": len(df),
    }
    
    return df, metadata


def inspect_file(file_path: str | Path) -> Dict[str, Any]:
    """
    Inspect a data file and return information about its structure.
    
    Useful for understanding file structure before creating column mappings.
    
    Args:
        file_path: Path to file to inspect
        
    Returns:
        Dict with file information including columns and sample values
    """
    df = read_file(file_path)
    
    columns_info = []
    for col in df.columns:
        non_null = df[col].notna().sum()
        sample_values = df[col].dropna().head(3).tolist()
        
        columns_info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null_count": non_null,
            "null_count": len(df) - non_null,
            "sample_values": sample_values,
        })
    
    # Try auto-detection
    mapper = ColumnMapper(df)
    suggestions = mapper.get_suggestions()
    
    return {
        "file_path": str(file_path),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_info,
        "auto_detected_mappings": suggestions,
        "unmapped_columns": mapper.get_unmapped_columns(),
    }

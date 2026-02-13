"""
Database management for SMART system.

Provides two main database classes:
- EmbeddingCache: Per-conference cache for embeddings with model/version tracking
- ConferenceDB: Working database for presentations, sessions, and placements
"""

import sqlite3
import hashlib
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import struct


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str
    model_version: str
    task_type: str = "SEMANTIC_SIMILARITY"
    dimensions: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task_type": self.task_type,
            "dimensions": self.dimensions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingConfig":
        return cls(**data)
    
    def config_hash(self) -> str:
        """Generate hash for this configuration."""
        config_str = f"{self.model_name}|{self.model_version}|{self.task_type}|{self.dimensions}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize numpy array to bytes for SQLite storage."""
    return embedding.astype(np.float32).tobytes()


def _deserialize_embedding(data: bytes, dimensions: int) -> np.ndarray:
    """Deserialize bytes back to numpy array."""
    return np.frombuffer(data, dtype=np.float32).reshape(-1)


class EmbeddingCache:
    """
    Per-conference embedding cache with model/version/task tracking.
    
    Stores embeddings keyed by text hash + configuration to enable:
    - Reuse of embeddings when text hasn't changed
    - Proper handling of model version changes
    - Different task type configurations (SEMANTIC_SIMILARITY, etc.)
    
    Usage:
        cache = EmbeddingCache("AIM2026_cache.db")
        
        # Check for existing embedding
        embedding = cache.get_embedding(text, config)
        if embedding is None:
            embedding = embedder.embed(text)
            cache.store_embedding(text, embedding, config)
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str | Path):
        """
        Initialize embedding cache.
        
        Args:
            db_path: Path to SQLite database file (created if doesn't exist)
        """
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            # Check schema version
            cursor = conn.execute(
                "SELECT value FROM cache_metadata WHERE key = 'schema_version'"
            )
            row = cursor.fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO cache_metadata (key, value) VALUES (?, ?)",
                    ("schema_version", str(self.SCHEMA_VERSION))
                )
                conn.execute(
                    "INSERT INTO cache_metadata (key, value) VALUES (?, ?)",
                    ("created_at", datetime.now().isoformat())
                )
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (text_hash, config_hash)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_model 
                ON embeddings(model_name, model_version)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    dimensions INTEGER,
                    first_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding_count INTEGER DEFAULT 0,
                    PRIMARY KEY (model_name, model_version, task_type)
                )
            """)
            
            conn.commit()
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate hash for text content."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get_embedding(
        self, 
        text: str, 
        config: EmbeddingConfig
    ) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding if available.
        
        Args:
            text: The text that was embedded
            config: Embedding configuration (model, version, task)
            
        Returns:
            numpy array of embedding if found, None otherwise
        """
        text_hash = self._hash_text(text)
        config_hash = config.config_hash()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT embedding, dimensions 
                FROM embeddings 
                WHERE text_hash = ? AND config_hash = ?
                """,
                (text_hash, config_hash)
            )
            row = cursor.fetchone()
            
            if row is not None:
                return _deserialize_embedding(row[0], row[1])
            return None
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        config: EmbeddingConfig
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Retrieve multiple cached embeddings at once.
        
        Args:
            texts: List of texts to look up
            config: Embedding configuration
            
        Returns:
            Dict mapping text to embedding (or None if not cached)
        """
        config_hash = config.config_hash()
        text_hashes = {self._hash_text(t): t for t in texts}
        
        results = {t: None for t in texts}
        
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(text_hashes))
            cursor = conn.execute(
                f"""
                SELECT text_hash, embedding, dimensions
                FROM embeddings
                WHERE text_hash IN ({placeholders}) AND config_hash = ?
                """,
                list(text_hashes.keys()) + [config_hash]
            )
            
            for row in cursor:
                text = text_hashes[row[0]]
                results[text] = _deserialize_embedding(row[1], row[2])
        
        return results
    
    def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        config: EmbeddingConfig
    ) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: The original text
            embedding: The embedding vector
            config: Embedding configuration used
        """
        text_hash = self._hash_text(text)
        config_hash = config.config_hash()
        dimensions = len(embedding)
        
        # Update config with actual dimensions
        config.dimensions = dimensions
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings 
                (text_hash, config_hash, text, embedding, model_name, model_version, 
                 task_type, dimensions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    text_hash,
                    config_hash,
                    text,
                    _serialize_embedding(embedding),
                    config.model_name,
                    config.model_version,
                    config.task_type,
                    dimensions,
                    datetime.now().isoformat()
                )
            )
            
            # Update model registry
            conn.execute(
                """
                INSERT INTO model_registry 
                (model_name, model_version, task_type, dimensions, embedding_count)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(model_name, model_version, task_type) DO UPDATE SET
                    last_used = CURRENT_TIMESTAMP,
                    embedding_count = embedding_count + 1,
                    dimensions = COALESCE(dimensions, excluded.dimensions)
                """,
                (config.model_name, config.model_version, config.task_type, dimensions)
            )
            
            conn.commit()
    
    def store_embeddings_batch(
        self,
        texts_and_embeddings: List[Tuple[str, np.ndarray]],
        config: EmbeddingConfig
    ) -> None:
        """
        Store multiple embeddings at once.
        
        Args:
            texts_and_embeddings: List of (text, embedding) tuples
            config: Embedding configuration used
        """
        if not texts_and_embeddings:
            return
            
        config_hash = config.config_hash()
        dimensions = len(texts_and_embeddings[0][1])
        config.dimensions = dimensions
        
        with sqlite3.connect(self.db_path) as conn:
            records = [
                (
                    self._hash_text(text),
                    config_hash,
                    text,
                    _serialize_embedding(embedding),
                    config.model_name,
                    config.model_version,
                    config.task_type,
                    len(embedding),
                    datetime.now().isoformat()
                )
                for text, embedding in texts_and_embeddings
            ]
            
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings 
                (text_hash, config_hash, text, embedding, model_name, model_version,
                 task_type, dimensions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )
            
            # Update model registry
            conn.execute(
                """
                INSERT INTO model_registry 
                (model_name, model_version, task_type, dimensions, embedding_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_name, model_version, task_type) DO UPDATE SET
                    last_used = CURRENT_TIMESTAMP,
                    embedding_count = embedding_count + excluded.embedding_count,
                    dimensions = COALESCE(dimensions, excluded.dimensions)
                """,
                (config.model_name, config.model_version, config.task_type, 
                 dimensions, len(texts_and_embeddings))
            )
            
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            total_embeddings = cursor.fetchone()[0]
            
            cursor = conn.execute(
                """
                SELECT model_name, model_version, task_type, dimensions, 
                       embedding_count, first_used, last_used
                FROM model_registry
                ORDER BY last_used DESC
                """
            )
            models = [
                {
                    "model_name": row[0],
                    "model_version": row[1],
                    "task_type": row[2],
                    "dimensions": row[3],
                    "embedding_count": row[4],
                    "first_used": row[5],
                    "last_used": row[6],
                }
                for row in cursor
            ]
            
            cursor = conn.execute(
                "SELECT value FROM cache_metadata WHERE key = 'created_at'"
            )
            row = cursor.fetchone()
            created_at = row[0] if row else None
            
        return {
            "total_embeddings": total_embeddings,
            "models": models,
            "created_at": created_at,
            "db_path": str(self.db_path),
        }
    
    def get_coverage_by_model(
        self, 
        texts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get cache coverage for a list of texts across all stored model configurations.
        
        This enables the UI to show which models have cached embeddings for the
        current presentations without needing to know the exact model version.
        
        Args:
            texts: List of texts to check coverage for
            
        Returns:
            List of dicts, each containing:
            - model_name: str
            - model_version: str
            - task_type: str
            - dimensions: int
            - cached_count: int (how many of the input texts are cached)
            - total_count: int (total embeddings in cache for this config)
            - coverage_pct: float (cached_count / len(texts) * 100)
            - config: EmbeddingConfig object for this model
        """
        if not texts:
            return []
        
        # Get hashes for the input texts
        text_hashes = [self._hash_text(t) for t in texts]
        text_hash_set = set(text_hashes)
        
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Get all unique model configurations from the registry
            cursor = conn.execute(
                """
                SELECT model_name, model_version, task_type, dimensions, embedding_count
                FROM model_registry
                ORDER BY last_used DESC
                """
            )
            model_configs = cursor.fetchall()
            
            for model_name, model_version, task_type, dimensions, total_count in model_configs:
                # Build config for this model
                config = EmbeddingConfig(
                    model_name=model_name,
                    model_version=model_version,
                    task_type=task_type,
                    dimensions=dimensions
                )
                config_hash = config.config_hash()
                
                # Count how many of the input texts are cached for this config
                placeholders = ",".join("?" * len(text_hashes))
                cursor = conn.execute(
                    f"""
                    SELECT COUNT(DISTINCT text_hash)
                    FROM embeddings
                    WHERE text_hash IN ({placeholders}) AND config_hash = ?
                    """,
                    text_hashes + [config_hash]
                )
                cached_count = cursor.fetchone()[0]
                
                # Only include configs that have at least one match
                if cached_count > 0:
                    results.append({
                        "model_name": model_name,
                        "model_version": model_version,
                        "task_type": task_type,
                        "dimensions": dimensions,
                        "cached_count": cached_count,
                        "total_count": total_count,
                        "coverage_pct": cached_count / len(texts) * 100,
                        "config": config,
                    })
        
        # Sort by coverage (highest first)
        results.sort(key=lambda x: x["cached_count"], reverse=True)
        return results
    
    def get_all_model_configs(self) -> List[Dict[str, Any]]:
        """
        Get all model configurations stored in the cache.
        
        Returns:
            List of dicts with model_name, model_version, task_type, dimensions,
            embedding_count, first_used, last_used, and config object.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT model_name, model_version, task_type, dimensions, 
                       embedding_count, first_used, last_used
                FROM model_registry
                ORDER BY last_used DESC
                """
            )
            
            results = []
            for row in cursor:
                config = EmbeddingConfig(
                    model_name=row[0],
                    model_version=row[1],
                    task_type=row[2],
                    dimensions=row[3]
                )
                results.append({
                    "model_name": row[0],
                    "model_version": row[1],
                    "task_type": row[2],
                    "dimensions": row[3],
                    "embedding_count": row[4],
                    "first_used": row[5],
                    "last_used": row[6],
                    "config": config,
                })
            
            return results


class ConferenceDB:
    """
    Working database for conference session organization.
    
    Stores presentations, sessions, placements, and generated content
    for a single conference. Supports incremental updates and tracks
    processing history.
    
    Usage:
        db = ConferenceDB("AIM2026_working.db", conference_year=26)
        
        # Import presentations
        db.import_presentations(df_presentations)
        
        # Create sessions
        db.create_session(session_id, title, presentation_ids)
        
        # Export for different purposes
        df = db.export_presentations()
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str | Path, conference_year: int = 26):
        """
        Initialize conference database.
        
        Args:
            db_path: Path to SQLite database file
            conference_year: Two-digit year for ID generation (default: 26)
        """
        self.db_path = Path(db_path)
        self.conference_year = conference_year
        self._temp_id_counter = 0
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conference_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            # Check/set schema version
            cursor = conn.execute(
                "SELECT value FROM conference_metadata WHERE key = 'schema_version'"
            )
            row = cursor.fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO conference_metadata (key, value) VALUES (?, ?)",
                    ("schema_version", str(self.SCHEMA_VERSION))
                )
                conn.execute(
                    "INSERT INTO conference_metadata (key, value) VALUES (?, ?)",
                    ("created_at", datetime.now().isoformat())
                )
                conn.execute(
                    "INSERT INTO conference_metadata (key, value) VALUES (?, ?)",
                    ("conference_year", str(self.conference_year))
                )
            
            # Presentations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS presentations (
                    abstract_id TEXT PRIMARY KEY,
                    is_temp_id INTEGER DEFAULT 0,
                    is_invited INTEGER DEFAULT 0,
                    source_submission_id TEXT,
                    source_abstract_id TEXT,
                    slot_number INTEGER DEFAULT 1,
                    title TEXT,
                    abstract TEXT,
                    combined_text TEXT,
                    presenter_first_name TEXT,
                    presenter_last_name TEXT,
                    presenter_email TEXT,
                    affiliation TEXT,
                    technical_community TEXT,
                    session_preference TEXT,
                    embedding_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    extra_data TEXT
                )
            """)
            
            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    title_options TEXT,
                    keywords TEXT,
                    coherence REAL,
                    distinctiveness REAL,
                    is_hybrid INTEGER DEFAULT 0,
                    placement_strategy TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    extra_data TEXT
                )
            """)
            
            # Placements table (many-to-many with history)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS placements (
                    placement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    abstract_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    placement_strategy TEXT,
                    presentation_fit REAL,
                    is_current INTEGER DEFAULT 1,
                    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (abstract_id) REFERENCES presentations(abstract_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_placements_current 
                ON placements(abstract_id, is_current)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_placements_session 
                ON placements(session_id, is_current)
            """)
            
            # Generated titles history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generated_titles (
                    generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    presentation_set_hash TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    title_1 TEXT,
                    title_2 TEXT,
                    title_3 TEXT,
                    keywords TEXT,
                    prompt_template TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Committee assignments
            conn.execute("""
                CREATE TABLE IF NOT EXISTS committees (
                    committee_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    combined_text TEXT,
                    embedding_hash TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_committee_matches (
                    session_id TEXT NOT NULL,
                    committee_id TEXT NOT NULL,
                    similarity_score REAL,
                    rank INTEGER,
                    PRIMARY KEY (session_id, committee_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (committee_id) REFERENCES committees(committee_id)
                )
            """)
            
            # Processing history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation TEXT NOT NULL,
                    parameters TEXT,
                    affected_count INTEGER,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            
            # Run migrations for existing databases
            self._run_migrations(conn)
    
    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """
        Run schema migrations for existing databases.
        
        Adds columns that may be missing from older database versions.
        """
        # Check existing columns in presentations table
        cursor = conn.execute("PRAGMA table_info(presentations)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        # Add source_abstract_id if missing (for multi-slot presentations)
        if "source_abstract_id" not in existing_columns:
            conn.execute("""
                ALTER TABLE presentations 
                ADD COLUMN source_abstract_id TEXT
            """)
        
        # Add slot_number if missing (for multi-slot presentations)
        if "slot_number" not in existing_columns:
            conn.execute("""
                ALTER TABLE presentations 
                ADD COLUMN slot_number INTEGER DEFAULT 1
            """)
        
        # Add is_invited if missing (for invited/hybrid presentations)
        if "is_invited" not in existing_columns:
            conn.execute("""
                ALTER TABLE presentations 
                ADD COLUMN is_invited INTEGER DEFAULT 0
            """)
        
        conn.commit()
    
    def generate_abstract_id(self, submission_id: Optional[str] = None) -> str:
        """
        Generate an abstract ID.
        
        Args:
            submission_id: Original submission ID if available
            
        Returns:
            Abstract ID in format YYXXXXX (real) or TEMP-YY-XXXXX (temporary)
        """
        year_prefix = str(self.conference_year).zfill(2)
        
        if submission_id:
            # Try to extract numeric portion
            numeric_part = ''.join(filter(str.isdigit, str(submission_id)))
            if numeric_part:
                # Use last 5 digits if longer
                numeric_part = numeric_part[-5:].zfill(5)
                return f"{year_prefix}{numeric_part}"
        
        # Generate temp ID
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM presentations WHERE is_temp_id = 1"
            )
            temp_count = cursor.fetchone()[0]
        
        self._temp_id_counter = max(self._temp_id_counter, temp_count) + 1
        return f"TEMP-{year_prefix}-{str(self._temp_id_counter).zfill(5)}"
    
    def import_presentation(
        self,
        title: str,
        abstract: Optional[str] = None,
        submission_id: Optional[str] = None,
        abstract_id: Optional[str] = None,
        source_abstract_id: Optional[str] = None,
        slot_number: int = 1,
        is_invited: bool = False,
        presenter_first_name: Optional[str] = None,
        presenter_last_name: Optional[str] = None,
        presenter_email: Optional[str] = None,
        affiliation: Optional[str] = None,
        technical_community: Optional[str] = None,
        session_preference: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Import a single presentation.
        
        Args:
            title: Presentation title (required)
            abstract: Presentation abstract
            submission_id: Original submission portal ID
            abstract_id: Pre-assigned abstract ID (if any)
            source_abstract_id: Original abstract ID before slot suffix (for multi-slot presentations)
            slot_number: Which time slot this occupies (1-based, for multi-slot presentations)
            is_invited: Whether this is an invited/hybrid presentation
            ... other optional fields
            
        Returns:
            The abstract_id assigned to this presentation
        """
        # Generate or validate abstract_id
        if abstract_id is None:
            abstract_id = self.generate_abstract_id(submission_id)
        
        # If no source_abstract_id provided, use the abstract_id
        if source_abstract_id is None:
            source_abstract_id = abstract_id
        
        is_temp = abstract_id.startswith("TEMP-")
        
        # Create combined text for embedding
        combined_text = f"{title}: {abstract}" if abstract else title
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO presentations
                (abstract_id, is_temp_id, is_invited, source_submission_id, source_abstract_id, slot_number,
                 title, abstract, combined_text, presenter_first_name, presenter_last_name,
                 presenter_email, affiliation, technical_community, session_preference,
                 updated_at, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    abstract_id,
                    1 if is_temp else 0,
                    1 if is_invited else 0,
                    submission_id,
                    source_abstract_id,
                    slot_number,
                    title,
                    abstract,
                    combined_text,
                    presenter_first_name,
                    presenter_last_name,
                    presenter_email,
                    affiliation,
                    technical_community,
                    session_preference,
                    datetime.now().isoformat(),
                    json.dumps(extra_data) if extra_data else None
                )
            )
            conn.commit()
        
        return abstract_id
    
    def import_presentations_batch(
        self,
        presentations: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Import multiple presentations at once.
        
        All columns from the source data are preserved. Known columns are stored
        in dedicated database fields, and any additional columns are stored in
        extra_data JSON for later export.
        
        Args:
            presentations: List of presentation dicts with keys matching
                          import_presentation parameters. Supports:
                          - abstract_id: Unique ID for this presentation slot
                          - source_abstract_id: Original ID for multi-slot presentations
                          - slot_number: Which slot (1-based) for multi-slot presentations
                          - is_invited: Whether this is an invited/hybrid presentation
                          - Any additional columns will be preserved in extra_data
                          
        Returns:
            List of assigned abstract_ids
        """
        # Define known columns that have dedicated database fields
        KNOWN_COLUMNS = {
            "abstract_id", "is_temp_id", "is_invited", "submission_id", 
            "source_submission_id", "source_abstract_id", "slot_number",
            "title", "abstract", "combined_text",
            "presenter_first_name", "presenter_last_name", "presenter_email",
            "affiliation", "technical_community", "session_preference",
            "embedding_hash", "created_at", "updated_at", "extra_data",
            # Also exclude internal/generated columns
            "session", "session_id", "presentation_fit",
        }
        
        abstract_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for pres in presentations:
                # Generate ID if needed
                abstract_id = pres.get("abstract_id")
                if abstract_id is None:
                    abstract_id = self.generate_abstract_id(pres.get("submission_id"))
                
                # Handle source_abstract_id for multi-slot presentations
                source_abstract_id = pres.get("source_abstract_id", abstract_id)
                slot_number = pres.get("slot_number", 1)
                is_invited = pres.get("is_invited", False)
                
                is_temp = abstract_id.startswith("TEMP-")
                
                title = pres.get("title", "")
                abstract = pres.get("abstract")
                combined_text = f"{title}: {abstract}" if abstract else title
                
                # Collect all unmapped columns into extra_data
                # Start with any existing extra_data
                extra_data = pres.get("extra_data")
                if isinstance(extra_data, str):
                    try:
                        extra_data = json.loads(extra_data)
                    except (json.JSONDecodeError, TypeError):
                        extra_data = {}
                elif extra_data is None:
                    extra_data = {}
                
                # Add any columns not in KNOWN_COLUMNS
                for key, value in pres.items():
                    if key not in KNOWN_COLUMNS and key not in extra_data:
                        # Convert non-serializable types
                        # Check for NaN (works for pandas NA and numpy nan)
                        try:
                            if value is None or (isinstance(value, float) and value != value):
                                extra_data[key] = None
                            elif hasattr(value, 'isoformat'):  # datetime
                                extra_data[key] = value.isoformat()
                            elif hasattr(value, 'item'):  # numpy scalar
                                extra_data[key] = value.item()
                            else:
                                extra_data[key] = value
                        except (TypeError, ValueError):
                            extra_data[key] = str(value) if value is not None else None
                
                conn.execute(
                    """
                    INSERT OR REPLACE INTO presentations
                    (abstract_id, is_temp_id, is_invited, source_submission_id, source_abstract_id, slot_number,
                     title, abstract, combined_text, presenter_first_name, presenter_last_name,
                     presenter_email, affiliation, technical_community, session_preference,
                     updated_at, extra_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        abstract_id,
                        1 if is_temp else 0,
                        1 if is_invited else 0,
                        pres.get("submission_id") or pres.get("source_submission_id"),
                        source_abstract_id,
                        slot_number,
                        title,
                        abstract,
                        combined_text,
                        pres.get("presenter_first_name"),
                        pres.get("presenter_last_name"),
                        pres.get("presenter_email"),
                        pres.get("affiliation"),
                        pres.get("technical_community"),
                        pres.get("session_preference"),
                        datetime.now().isoformat(),
                        json.dumps(extra_data) if extra_data else None
                    )
                )
                abstract_ids.append(abstract_id)
            
            # Log operation
            conn.execute(
                """
                INSERT INTO processing_history (operation, parameters, affected_count)
                VALUES (?, ?, ?)
                """,
                ("import_presentations", None, len(presentations))
            )
            
            conn.commit()
        
        return abstract_ids
    
    def get_presentations(
        self,
        include_temp: bool = True,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get presentations from database.
        
        Args:
            include_temp: Whether to include temporary ID presentations
            session_id: Filter to specific session (if assigned)
            
        Returns:
            List of presentation dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if session_id:
                cursor = conn.execute(
                    """
                    SELECT p.*, pl.session_id, pl.presentation_fit
                    FROM presentations p
                    JOIN placements pl ON p.abstract_id = pl.abstract_id
                    WHERE pl.session_id = ? AND pl.is_current = 1
                    ORDER BY pl.presentation_fit DESC
                    """,
                    (session_id,)
                )
            else:
                query = "SELECT * FROM presentations"
                if not include_temp:
                    query += " WHERE is_temp_id = 0"
                cursor = conn.execute(query)
            
            return [dict(row) for row in cursor]
    
    def get_presentation(self, abstract_id: str) -> Optional[Dict[str, Any]]:
        """Get a single presentation by ID, including current session assignment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT p.*, pl.session_id, pl.presentation_fit
                FROM presentations p
                LEFT JOIN placements pl ON p.abstract_id = pl.abstract_id AND pl.is_current = 1
                WHERE p.abstract_id = ?
                """,
                (abstract_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_embedding_hash(self, abstract_id: str, embedding_hash: str) -> None:
        """Update the embedding hash for a presentation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE presentations 
                SET embedding_hash = ?, updated_at = ?
                WHERE abstract_id = ?
                """,
                (embedding_hash, datetime.now().isoformat(), abstract_id)
            )
            conn.commit()
    
    def create_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        presentation_ids: Optional[List[str]] = None,
        is_hybrid: bool = False,
        placement_strategy: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier
            title: Session title
            presentation_ids: List of abstract_ids to assign
            is_hybrid: Whether this is a hybrid/invited session
            placement_strategy: Name of algorithm used for placement
            extra_data: Additional session data
            
        Returns:
            The session_id
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, title, is_hybrid, placement_strategy, updated_at, extra_data)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    title,
                    1 if is_hybrid else 0,
                    placement_strategy,
                    datetime.now().isoformat(),
                    json.dumps(extra_data) if extra_data else None
                )
            )
            
            if presentation_ids:
                # Mark old placements as not current
                conn.execute(
                    """
                    UPDATE placements SET is_current = 0
                    WHERE session_id = ? AND is_current = 1
                    """,
                    (session_id,)
                )
                
                # Add new placements
                for abstract_id in presentation_ids:
                    conn.execute(
                        """
                        INSERT INTO placements 
                        (abstract_id, session_id, placement_strategy, is_current)
                        VALUES (?, ?, ?, 1)
                        """,
                        (abstract_id, session_id, placement_strategy)
                    )
            
            conn.commit()
        
        return session_id
    
    def get_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions with presentation counts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT s.*, COUNT(pl.abstract_id) as presentation_count
                FROM sessions s
                LEFT JOIN placements pl ON s.session_id = pl.session_id AND pl.is_current = 1
                GROUP BY s.session_id
                ORDER BY s.session_id
                """
            )
            return [dict(row) for row in cursor]
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a single session with its presentations."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            session_row = cursor.fetchone()
            if not session_row:
                return None
            
            session = dict(session_row)
            
            # Get presentations
            cursor = conn.execute(
                """
                SELECT p.*, pl.presentation_fit
                FROM presentations p
                JOIN placements pl ON p.abstract_id = pl.abstract_id
                WHERE pl.session_id = ? AND pl.is_current = 1
                ORDER BY pl.presentation_fit DESC
                """,
                (session_id,)
            )
            session["presentations"] = [dict(row) for row in cursor]
            
            return session
    
    def update_session_metrics(
        self,
        session_id: str,
        coherence: float,
        distinctiveness: Optional[float] = None
    ) -> None:
        """Update computed metrics for a session."""
        # Convert numpy scalars to Python floats to avoid SQLite serialization issues
        if hasattr(coherence, 'item'):
            coherence = coherence.item()
        elif coherence is not None:
            coherence = float(coherence)
            
        if hasattr(distinctiveness, 'item'):
            distinctiveness = distinctiveness.item()
        elif distinctiveness is not None:
            distinctiveness = float(distinctiveness)
            
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE sessions 
                SET coherence = ?, distinctiveness = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (coherence, distinctiveness, datetime.now().isoformat(), session_id)
            )
            conn.commit()
    
    def update_session_titles(
        self,
        session_id: str,
        title_options: List[str],
        keywords: List[str],
        model_name: str,
        selected_title: Optional[str] = None
    ) -> None:
        """
        Update generated titles for a session.
        
        Args:
            session_id: Session to update
            title_options: List of generated title options
            keywords: List of generated keywords
            model_name: LLM used for generation
            selected_title: Which title to use (defaults to first option)
        """
        # Compute hash of current presentation set
        presentations = self.get_presentations(session_id=session_id)
        pres_ids = sorted([p["abstract_id"] for p in presentations])
        pres_set_hash = hashlib.sha256(
            "|".join(pres_ids).encode()
        ).hexdigest()[:16]
        
        with sqlite3.connect(self.db_path) as conn:
            # Store in history
            conn.execute(
                """
                INSERT INTO generated_titles
                (session_id, presentation_set_hash, model_name, 
                 title_1, title_2, title_3, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    pres_set_hash,
                    model_name,
                    title_options[0] if len(title_options) > 0 else None,
                    title_options[1] if len(title_options) > 1 else None,
                    title_options[2] if len(title_options) > 2 else None,
                    ", ".join(keywords)
                )
            )
            
            # Update session
            selected = selected_title or (title_options[0] if title_options else None)
            conn.execute(
                """
                UPDATE sessions
                SET title = ?, title_options = ?, keywords = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    selected,
                    json.dumps(title_options),
                    ", ".join(keywords),
                    datetime.now().isoformat(),
                    session_id
                )
            )
            
            conn.commit()
    
    def clear_presentations(self, keep_sessions: bool = False) -> int:
        """
        Clear all presentations from the database.
        
        Allows reloading a new set of presentations without dropping
        the conference. Embeddings in the EmbeddingCache are preserved
        (they're in a separate database).
        
        Args:
            keep_sessions: If True, keep session definitions (clear placements only)
            
        Returns:
            Number of presentations removed
        """
        with sqlite3.connect(self.db_path) as conn:
            # Count presentations being removed
            cursor = conn.execute("SELECT COUNT(*) FROM presentations")
            count = cursor.fetchone()[0]
            
            # Clear placements first (foreign key constraint)
            conn.execute("DELETE FROM placements")
            
            # Clear presentations
            conn.execute("DELETE FROM presentations")
            
            if not keep_sessions:
                # Clear sessions and related tables
                conn.execute("DELETE FROM sessions")
                conn.execute("DELETE FROM generated_titles")
                conn.execute("DELETE FROM session_committee_matches")
            
            # Log operation
            conn.execute(
                """
                INSERT INTO processing_history (operation, parameters, affected_count)
                VALUES (?, ?, ?)
                """,
                ("clear_presentations", json.dumps({"keep_sessions": keep_sessions}), count)
            )
            
            conn.commit()
            
        return count
    
    def clear_sessions(self) -> int:
        """
        Clear all sessions and placements, keeping presentations.
        
        Useful for re-running placement with different parameters.
        
        Returns:
            Number of sessions removed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            count = cursor.fetchone()[0]
            
            conn.execute("DELETE FROM placements")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM generated_titles")
            conn.execute("DELETE FROM session_committee_matches")
            
            # Log operation
            conn.execute(
                """
                INSERT INTO processing_history (operation, parameters, affected_count)
                VALUES (?, ?, ?)
                """,
                ("clear_sessions", None, count)
            )
            
            conn.commit()
            
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM presentations")
            total_presentations = cursor.fetchone()[0]
            
            cursor = conn.execute(
                "SELECT COUNT(*) FROM presentations WHERE is_temp_id = 1"
            )
            temp_presentations = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT abstract_id) FROM placements WHERE is_current = 1"
            )
            placed_presentations = cursor.fetchone()[0]
            
            cursor = conn.execute(
                "SELECT value FROM conference_metadata WHERE key = 'created_at'"
            )
            row = cursor.fetchone()
            created_at = row[0] if row else None
            
        return {
            "total_presentations": total_presentations,
            "temp_presentations": temp_presentations,
            "real_presentations": total_presentations - temp_presentations,
            "total_sessions": total_sessions,
            "placed_presentations": placed_presentations,
            "unplaced_presentations": total_presentations - placed_presentations,
            "created_at": created_at,
            "db_path": str(self.db_path),
        }
    
    def export_to_dataframe(
        self, 
        include_embeddings: bool = False,
        flatten_extra_data: bool = True,
        include_session_info: bool = True,
    ):
        """
        Export presentations to pandas DataFrame.
        
        Reconstructs the original spreadsheet data by flattening extra_data
        columns back to top-level columns. This allows users to get back their
        original data plus session assignments without manual merging.
        
        Args:
            include_embeddings: Whether to include embedding data
            flatten_extra_data: Whether to expand extra_data JSON into columns
            include_session_info: Whether to add session_id and fit columns
            
        Returns:
            pandas DataFrame with presentation data and all original columns
        """
        import pandas as pd
        
        presentations = self.get_presentations()
        df = pd.DataFrame(presentations)
        
        # Flatten extra_data into separate columns
        if flatten_extra_data and "extra_data" in df.columns:
            extra_rows = []
            for _, row in df.iterrows():
                extra = row.get("extra_data")
                if extra:
                    if isinstance(extra, str):
                        try:
                            extra = json.loads(extra)
                        except (json.JSONDecodeError, TypeError):
                            extra = {}
                    extra_rows.append(extra if isinstance(extra, dict) else {})
                else:
                    extra_rows.append({})
            
            # Create DataFrame from extra_data and merge
            if extra_rows:
                extra_df = pd.DataFrame(extra_rows)
                # Don't overwrite existing columns
                new_cols = [c for c in extra_df.columns if c not in df.columns]
                if new_cols:
                    for col in new_cols:
                        df[col] = extra_df[col]
            
            # Optionally drop the extra_data column after flattening
            df = df.drop(columns=["extra_data"], errors="ignore")
        
        # Add session assignment
        if include_session_info:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT abstract_id, session_id, presentation_fit
                    FROM placements WHERE is_current = 1
                    """
                )
                placements = {row[0]: (row[1], row[2]) for row in cursor}
            
            df["session_id"] = df["abstract_id"].map(
                lambda x: placements.get(x, (None, None))[0]
            )
            df["presentation_fit"] = df["abstract_id"].map(
                lambda x: placements.get(x, (None, None))[1]
            )
        
        # Clean up internal columns that users don't need
        internal_cols = ["combined_text", "embedding_hash", "is_temp_id", "updated_at", "created_at"]
        df = df.drop(columns=[c for c in internal_cols if c in df.columns], errors="ignore")
        
        return df

    # ========== Committee Methods ==========
    
    def import_committee(
        self,
        committee_id: str,
        name: str,
        description: Optional[str] = None,
        combined_text: Optional[str] = None,
        embedding_hash: Optional[str] = None,
    ) -> str:
        """
        Import a single committee.
        
        Args:
            committee_id: Unique committee identifier
            name: Committee name
            description: Committee description
            combined_text: Pre-computed combined text for embedding
            embedding_hash: Hash of the embedding if already computed
            
        Returns:
            The committee_id
        """
        if combined_text is None and description:
            combined_text = f"{name}: {description}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO committees
                (committee_id, name, description, combined_text, embedding_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (committee_id, name, description, combined_text, embedding_hash)
            )
            conn.commit()
        
        return committee_id
    
    def import_committees_batch(
        self,
        committees: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Import multiple committees efficiently.
        
        Args:
            committees: List of committee dicts with keys:
                - committee_id (optional, auto-generated if missing)
                - name or committee_name (required)
                - description (optional)
                - combined_text (optional)
                
        Returns:
            List of imported committee_ids
        """
        imported_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for i, committee in enumerate(committees):
                # Handle various key names
                committee_id = committee.get("committee_id") or f"COM-{i+1:03d}"
                name = committee.get("name") or committee.get("committee_name", "")
                description = committee.get("description", "")
                combined_text = committee.get("combined_text")
                
                if combined_text is None and description:
                    combined_text = f"{name}: {description}"
                
                conn.execute(
                    """
                    INSERT OR REPLACE INTO committees
                    (committee_id, name, description, combined_text, embedding_hash)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (committee_id, name, description, combined_text, None)
                )
                imported_ids.append(committee_id)
            
            conn.commit()
            
            # Log to processing history
            conn.execute(
                """
                INSERT INTO processing_history (operation, parameters, affected_count)
                VALUES (?, ?, ?)
                """,
                ("import_committees_batch", None, len(imported_ids))
            )
            conn.commit()
        
        return imported_ids
    
    def get_committees(self) -> List[Dict[str, Any]]:
        """Get all committees."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT committee_id, name, description, combined_text, embedding_hash
                FROM committees
                ORDER BY committee_id
                """
            )
            return [dict(row) for row in cursor]
    
    def get_committee(self, committee_id: str) -> Optional[Dict[str, Any]]:
        """Get a single committee by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM committees WHERE committee_id = ?",
                (committee_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_committee_embedding_hash(
        self,
        committee_id: str,
        embedding_hash: str,
    ) -> None:
        """Update the embedding hash for a committee."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE committees SET embedding_hash = ? WHERE committee_id = ?",
                (embedding_hash, committee_id)
            )
            conn.commit()
    
    def clear_committees(self) -> int:
        """
        Remove all committees.
        
        Returns:
            Number of committees removed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM committees")
            count = cursor.fetchone()[0]
            
            conn.execute("DELETE FROM session_committee_matches")
            conn.execute("DELETE FROM committees")
            
            conn.execute(
                """
                INSERT INTO processing_history (operation, parameters, affected_count)
                VALUES (?, ?, ?)
                """,
                ("clear_committees", None, count)
            )
            conn.commit()
        
        return count
    
    def delete_committee(self, committee_id: str) -> bool:
        """
        Delete a single committee.
        
        Args:
            committee_id: ID of committee to delete
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM session_committee_matches WHERE committee_id = ?",
                (committee_id,)
            )
            cursor = conn.execute(
                "DELETE FROM committees WHERE committee_id = ?",
                (committee_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    # ========== Session-Committee Match Methods ==========
    
    def store_session_committee_matches(
        self,
        session_id: str,
        matches: List[Tuple[str, float, int]],
    ) -> None:
        """
        Store committee matches for a session.
        
        Args:
            session_id: Session ID
            matches: List of (committee_id, similarity_score, rank) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing matches for this session
            conn.execute(
                "DELETE FROM session_committee_matches WHERE session_id = ?",
                (session_id,)
            )
            
            # Insert new matches
            for committee_id, score, rank in matches:
                conn.execute(
                    """
                    INSERT INTO session_committee_matches
                    (session_id, committee_id, similarity_score, rank)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, committee_id, score, rank)
                )
            
            conn.commit()
    
    def get_session_committee_matches(
        self,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get committee matches for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List of match dicts with committee info and scores
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT scm.*, c.name as committee_name, c.description
                FROM session_committee_matches scm
                JOIN committees c ON scm.committee_id = c.committee_id
                WHERE scm.session_id = ?
                ORDER BY scm.rank
                """,
                (session_id,)
            )
            return [dict(row) for row in cursor]
    
    def get_committee_sessions(
        self,
        committee_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions matched to a committee.
        
        Args:
            committee_id: Committee ID
            
        Returns:
            List of session matches
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT scm.*, s.title as session_title
                FROM session_committee_matches scm
                JOIN sessions s ON scm.session_id = s.session_id
                WHERE scm.committee_id = ?
                ORDER BY scm.similarity_score DESC
                """,
                (committee_id,)
            )
            return [dict(row) for row in cursor]
    
    def clear_session_committee_matches(
        self,
        session_id: Optional[str] = None,
    ) -> int:
        """
        Clear committee matches.
        
        Args:
            session_id: If provided, only clear matches for this session.
                       If None, clear all matches.
                       
        Returns:
            Number of matches removed
        """
        with sqlite3.connect(self.db_path) as conn:
            if session_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM session_committee_matches WHERE session_id = ?",
                    (session_id,)
                )
                count = cursor.fetchone()[0]
                conn.execute(
                    "DELETE FROM session_committee_matches WHERE session_id = ?",
                    (session_id,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM session_committee_matches")
                count = cursor.fetchone()[0]
                conn.execute("DELETE FROM session_committee_matches")
            
            conn.commit()
        
        return count
    
    # ========== Placement Utility Methods ==========
    
    def get_unplaced_presentations(self) -> List[Dict[str, Any]]:
        """
        Get all presentations not currently assigned to a session.
        
        Returns:
            List of presentation dicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT p.*
                FROM presentations p
                LEFT JOIN placements pl ON p.abstract_id = pl.abstract_id AND pl.is_current = 1
                WHERE pl.abstract_id IS NULL
                ORDER BY p.abstract_id
                """
            )
            return [dict(row) for row in cursor]
    
    def add_presentation_to_session(
        self,
        abstract_id: str,
        session_id: str,
        fit_score: Optional[float] = None,
        strategy: Optional[str] = None,
    ) -> None:
        """
        Add a single presentation to a session.
        
        Args:
            abstract_id: Presentation ID
            session_id: Session ID
            fit_score: Optional fit score
            strategy: Optional placement strategy name
        """
        with sqlite3.connect(self.db_path) as conn:
            # Mark any existing current placement as not current
            conn.execute(
                """
                UPDATE placements SET is_current = 0
                WHERE abstract_id = ? AND is_current = 1
                """,
                (abstract_id,)
            )
            
            # Add new placement
            conn.execute(
                """
                INSERT INTO placements
                (abstract_id, session_id, placement_strategy, presentation_fit, is_current)
                VALUES (?, ?, ?, ?, 1)
                """,
                (abstract_id, session_id, strategy, fit_score)
            )
            
            conn.commit()
    
    def remove_presentation_from_session(
        self,
        abstract_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Remove a presentation from its current session.
        Also clears metrics on the affected session.
        
        Args:
            abstract_id: Presentation ID
            session_id: If provided, only remove from this specific session
            
        Returns:
            True if removed, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get affected session(s) for metric clearing
            if session_id:
                affected_sessions = [session_id]
                cursor = conn.execute(
                    """
                    UPDATE placements SET is_current = 0
                    WHERE abstract_id = ? AND session_id = ? AND is_current = 1
                    """,
                    (abstract_id, session_id)
                )
            else:
                # Find which sessions this presentation is in
                sess_cursor = conn.execute(
                    "SELECT DISTINCT session_id FROM placements WHERE abstract_id = ? AND is_current = 1",
                    (abstract_id,)
                )
                affected_sessions = [row[0] for row in sess_cursor.fetchall()]
                
                cursor = conn.execute(
                    """
                    UPDATE placements SET is_current = 0
                    WHERE abstract_id = ? AND is_current = 1
                    """,
                    (abstract_id,)
                )
            
            # Clear metrics on affected sessions
            if cursor.rowcount > 0 and affected_sessions:
                placeholders = ','.join(['?'] * len(affected_sessions))
                conn.execute(
                    f"""
                    UPDATE sessions SET coherence = NULL, distinctiveness = NULL, updated_at = ?
                    WHERE session_id IN ({placeholders})
                    """,
                    [datetime.now().isoformat()] + affected_sessions
                )
            
            conn.commit()
            return cursor.rowcount > 0
    
    def move_presentation(
        self,
        abstract_id: str,
        to_session_id: str,
        fit_score: Optional[float] = None,
    ) -> None:
        """
        Move a presentation from its current session to another.
        Clears metrics on both source and destination sessions.
        
        Args:
            abstract_id: Presentation ID
            to_session_id: Destination session ID
            fit_score: Optional new fit score
        """
        self.remove_presentation_from_session(abstract_id)  # This now clears source session metrics
        self.add_presentation_to_session(abstract_id, to_session_id, fit_score, "manual_move")
        # Clear metrics on destination session
        self.clear_session_metrics(to_session_id)
    
    def get_placements(
        self,
        session_id: Optional[str] = None,
        current_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get placement records.
        
        Args:
            session_id: If provided, only get placements for this session
            current_only: If True, only get current placements
            
        Returns:
            List of placement dicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM placements WHERE 1=1"
            params = []
            
            if current_only:
                query += " AND is_current = 1"
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            query += " ORDER BY session_id, abstract_id"
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor]
    
    def update_placement_fit(
        self,
        abstract_id: str,
        session_id: str,
        fit_score: float,
    ) -> bool:
        """
        Update the fit score for a placement.
        
        Args:
            abstract_id: Presentation ID
            session_id: Session ID
            fit_score: New fit score
            
        Returns:
            True if updated, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE placements SET presentation_fit = ?
                WHERE abstract_id = ? AND session_id = ? AND is_current = 1
                """,
                (fit_score, abstract_id, session_id)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def update_presentation(
        self,
        abstract_id: str,
        title: Optional[str] = None,
        abstract: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Update presentation fields.
        
        Args:
            abstract_id: Presentation ID to update
            title: New title (if provided)
            abstract: New abstract (if provided)
            **kwargs: Additional fields to update
            
        Returns:
            True if updated, False if not found
        """
        updates = []
        params = []
        
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        
        if abstract is not None:
            updates.append("abstract = ?")
            params.append(abstract)
        
        # Handle kwargs for flexibility
        for key, value in kwargs.items():
            if key in ("technical_community", "session_preference", "presenter_email",
                       "presenter_first_name", "presenter_last_name", "affiliation"):
                updates.append(f"{key} = ?")
                params.append(value)
        
        if not updates:
            return False
        
        params.append(abstract_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE presentations SET {', '.join(updates)} WHERE abstract_id = ?",
                params
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def clear_session_metrics(self, session_id: Optional[str] = None) -> int:
        """
        Clear coherence and distinctiveness metrics for session(s).
        
        Args:
            session_id: Specific session to clear, or None for all sessions
            
        Returns:
            Number of sessions affected
        """
        with sqlite3.connect(self.db_path) as conn:
            if session_id:
                cursor = conn.execute(
                    """
                    UPDATE sessions SET coherence = NULL, distinctiveness = NULL, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (datetime.now().isoformat(), session_id)
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE sessions SET coherence = NULL, distinctiveness = NULL, updated_at = ?
                    """,
                    (datetime.now().isoformat(),)
                )
            conn.commit()
            return cursor.rowcount
    
    def get_sessions_needing_metrics(self) -> List[str]:
        """
        Get list of session IDs that have NULL coherence (need recalculation).
        
        Returns:
            List of session IDs needing metric calculation
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT session_id FROM sessions WHERE coherence IS NULL"
            )
            return [row[0] for row in cursor.fetchall()]

    def delete_presentation(self, abstract_id: str) -> bool:
        """
        Delete a single presentation.
        Also clears metrics on any affected sessions.
        
        Args:
            abstract_id: ID of presentation to delete
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get affected session IDs before deleting placements
            cursor = conn.execute(
                "SELECT DISTINCT session_id FROM placements WHERE abstract_id = ? AND is_current = 1",
                (abstract_id,)
            )
            affected_sessions = [row[0] for row in cursor.fetchall()]
            
            # Remove placements
            conn.execute(
                "DELETE FROM placements WHERE abstract_id = ?",
                (abstract_id,)
            )
            
            # Clear metrics on affected sessions
            if affected_sessions:
                placeholders = ','.join(['?'] * len(affected_sessions))
                conn.execute(
                    f"""
                    UPDATE sessions SET coherence = NULL, distinctiveness = NULL, updated_at = ?
                    WHERE session_id IN ({placeholders})
                    """,
                    [datetime.now().isoformat()] + affected_sessions
                )
            
            cursor = conn.execute(
                "DELETE FROM presentations WHERE abstract_id = ?",
                (abstract_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a single session.
        
        Args:
            session_id: ID of session to delete
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            # Remove placements first
            conn.execute(
                "UPDATE placements SET is_current = 0 WHERE session_id = ?",
                (session_id,)
            )
            
            # Remove committee matches
            conn.execute(
                "DELETE FROM session_committee_matches WHERE session_id = ?",
                (session_id,)
            )
            
            # Remove generated titles
            conn.execute(
                "DELETE FROM generated_titles WHERE session_id = ?",
                (session_id,)
            )
            
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        is_hybrid: Optional[bool] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Update session fields.
        
        Args:
            session_id: Session ID to update
            title: New title (if provided)
            is_hybrid: New hybrid status (if provided)
            extra_data: New extra data (if provided)
            **kwargs: Additional fields to update
            
        Returns:
            True if updated, False if not found
        """
        updates = []
        params = []
        
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        
        if is_hybrid is not None:
            updates.append("is_hybrid = ?")
            params.append(1 if is_hybrid else 0)
        
        if extra_data is not None:
            updates.append("extra_data = ?")
            params.append(json.dumps(extra_data))
        
        # Handle kwargs for flexibility
        for key, value in kwargs.items():
            if key in ("coherence", "distinctiveness", "placement_strategy"):
                updates.append(f"{key} = ?")
                params.append(value)
        
        if not updates:
            return False
        
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(session_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?",
                params
            )
            conn.commit()
            return cursor.rowcount > 0

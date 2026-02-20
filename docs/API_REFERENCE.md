# SMART Package — API Reference

> Auto-generated from source on 2026-02-19.

---

## Table of Contents

1. [smart.core.database](#smartcoredatabase)
2. [smart.core.placement](#smartcoreplacement)
3. [smart.core.metrics](#smartcoremetrics)
4. [smart.io.loaders](#smartioloaders)
5. [smart.io.exporters](#smartioexporters)
6. [smart.llm.embeddings](#smartllmembeddings)
7. [smart.llm.titles](#smartllmtitles)

---

## `smart.core.database`

> Database management for SMART system.

### Constants

| Name | Type | Value |
|------|------|-------|
| `EmbeddingCache.SCHEMA_VERSION` | `int` | `1` |
| `ConferenceDB.SCHEMA_VERSION` | `int` | `1` |

---

### `@dataclass` EmbeddingConfig

Configuration for embedding generation.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `model_name` | `str` | *(required)* |
| `model_version` | `str` | *(required)* |
| `task_type` | `str` | `"SEMANTIC_SIMILARITY"` |
| `dimensions` | `Optional[int]` | `None` |

#### Methods

```python
def to_dict(self) -> Dict[str, Any]
```
Serialize configuration to a dictionary.

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingConfig"
```
Deserialize configuration from a dictionary.

```python
def config_hash(self) -> str
```
Generate hash for this configuration.

---

### Class `EmbeddingCache`

Per-conference embedding cache with model/version/task tracking.

#### Constructor

```python
def __init__(self, db_path: str | Path)
```
Initialize embedding cache.

#### Methods

```python
def get_embedding(
    self,
    text: str,
    config: EmbeddingConfig
) -> Optional[np.ndarray]
```
Retrieve cached embedding if available.

```python
def get_embeddings_batch(
    self,
    texts: List[str],
    config: EmbeddingConfig
) -> Dict[str, Optional[np.ndarray]]
```
Retrieve multiple cached embeddings at once.

```python
def store_embedding(
    self,
    text: str,
    embedding: np.ndarray,
    config: EmbeddingConfig
) -> None
```
Store embedding in cache.

```python
def store_embeddings_batch(
    self,
    texts_and_embeddings: List[Tuple[str, np.ndarray]],
    config: EmbeddingConfig
) -> None
```
Store multiple embeddings at once.

```python
def get_stats(self) -> Dict[str, Any]
```
Get cache statistics.

```python
def get_coverage_by_model(
    self,
    texts: List[str]
) -> List[Dict[str, Any]]
```
Get cache coverage for a list of texts across all stored model configurations.

```python
def get_all_model_configs(self) -> List[Dict[str, Any]]
```
Get all model configurations stored in the cache.

---

### Class `ConferenceDB`

Working database for conference session organization.

#### Constructor

```python
def __init__(self, db_path: str | Path, conference_year: int = 26)
```
Initialize conference database.

#### Methods — Presentations

```python
def generate_abstract_id(self, submission_id: Optional[str] = None) -> str
```
Generate an abstract ID.

```python
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
) -> str
```
Import a single presentation.

```python
def import_presentations_batch(
    self,
    presentations: List[Dict[str, Any]]
) -> List[str]
```
Import multiple presentations at once.

```python
def get_presentations(
    self,
    include_temp: bool = True,
    session_id: Optional[str] = None
) -> List[Dict[str, Any]]
```
Get presentations from database.

```python
def get_presentation(self, abstract_id: str) -> Optional[Dict[str, Any]]
```
Get a single presentation by ID, including current session assignment.

```python
def update_embedding_hash(self, abstract_id: str, embedding_hash: str) -> None
```
Update the embedding hash for a presentation.

```python
def update_presentation(
    self,
    abstract_id: str,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
    **kwargs
) -> bool
```
Update presentation fields.

```python
def delete_presentation(self, abstract_id: str) -> bool
```
Delete a single presentation.

```python
def clear_presentations(self, keep_sessions: bool = False) -> int
```
Clear all presentations from the database.

```python
def get_unplaced_presentations(self) -> List[Dict[str, Any]]
```
Get all presentations not currently assigned to a session.

```python
def export_to_dataframe(
    self,
    include_embeddings: bool = False,
    flatten_extra_data: bool = True,
    include_session_info: bool = True,
)
```
Export presentations to pandas DataFrame.

#### Methods — Sessions

```python
def create_session(
    self,
    session_id: str,
    title: Optional[str] = None,
    presentation_ids: Optional[List[str]] = None,
    is_hybrid: bool = False,
    placement_strategy: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> str
```
Create a new session.

```python
def get_sessions(self) -> List[Dict[str, Any]]
```
Get all sessions with presentation counts.

```python
def get_session(self, session_id: str) -> Optional[Dict[str, Any]]
```
Get a single session with its presentations.

```python
def update_session_metrics(
    self,
    session_id: str,
    coherence: float,
    distinctiveness: Optional[float] = None
) -> None
```
Update computed metrics for a session.

```python
def update_session_titles(
    self,
    session_id: str,
    title_options: List[str],
    keywords: List[str],
    model_name: str,
    selected_title: Optional[str] = None
) -> None
```
Update generated titles for a session.

```python
def update_session(
    self,
    session_id: str,
    title: Optional[str] = None,
    is_hybrid: Optional[bool] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> bool
```
Update session fields.

```python
def delete_session(self, session_id: str) -> bool
```
Delete a single session.

```python
def clear_sessions(self) -> int
```
Clear all sessions and placements, keeping presentations.

```python
def clear_session_metrics(self, session_id: Optional[str] = None) -> int
```
Clear coherence and distinctiveness metrics for session(s).

```python
def get_sessions_needing_metrics(self) -> List[str]
```
Get list of session IDs that have NULL coherence (need recalculation).

#### Methods — Placements

```python
def add_presentation_to_session(
    self,
    abstract_id: str,
    session_id: str,
    fit_score: Optional[float] = None,
    strategy: Optional[str] = None,
) -> None
```
Add a single presentation to a session.

```python
def remove_presentation_from_session(
    self,
    abstract_id: str,
    session_id: Optional[str] = None,
) -> bool
```
Remove a presentation from its current session.

```python
def move_presentation(
    self,
    abstract_id: str,
    to_session_id: str,
    fit_score: Optional[float] = None,
) -> None
```
Move a presentation from its current session to another.

```python
def get_placements(
    self,
    session_id: Optional[str] = None,
    current_only: bool = True,
) -> List[Dict[str, Any]]
```
Get placement records.

```python
def update_placement_fit(
    self,
    abstract_id: str,
    session_id: str,
    fit_score: float,
) -> bool
```
Update the fit score for a placement.

#### Methods — Committees

```python
def import_committee(
    self,
    committee_id: str,
    name: str,
    description: Optional[str] = None,
    combined_text: Optional[str] = None,
    embedding_hash: Optional[str] = None,
) -> str
```
Import a single committee.

```python
def import_committees_batch(
    self,
    committees: List[Dict[str, Any]],
) -> List[str]
```
Import multiple committees efficiently.

```python
def get_committees(self) -> List[Dict[str, Any]]
```
Get all committees.

```python
def get_committee(self, committee_id: str) -> Optional[Dict[str, Any]]
```
Get a single committee by ID.

```python
def update_committee_embedding_hash(
    self,
    committee_id: str,
    embedding_hash: str,
) -> None
```
Update the embedding hash for a committee.

```python
def clear_committees(self) -> int
```
Remove all committees.

```python
def delete_committee(self, committee_id: str) -> bool
```
Delete a single committee.

#### Methods — Session-Committee Matches

```python
def store_session_committee_matches(
    self,
    session_id: str,
    matches: List[Tuple[str, float, int]],
) -> None
```
Store committee matches for a session.

```python
def get_session_committee_matches(
    self,
    session_id: str,
) -> List[Dict[str, Any]]
```
Get committee matches for a session.

```python
def get_committee_sessions(
    self,
    committee_id: str,
) -> List[Dict[str, Any]]
```
Get all sessions matched to a committee.

```python
def clear_session_committee_matches(
    self,
    session_id: Optional[str] = None,
) -> int
```
Clear committee matches.

#### Methods — Statistics

```python
def get_stats(self) -> Dict[str, Any]
```
Get database statistics.

---

## `smart.core.placement`

> Placement strategies for session organization.

---

### `@dataclass` PlacementResult

Result of a placement operation.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `session_assignments` | `Dict[str, str]` | *(required)* |
| `sessions` | `List[Dict[str, Any]]` | *(required)* |
| `metadata` | `Dict[str, Any]` | *(required)* |
| `unassigned` | `List[str]` | `field(default_factory=list)` |

---

### `@dataclass` SessionConstraints

Constraints for session creation.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `min_session_size` | `int` | `8` |
| `max_session_size` | `int` | `12` |
| `max_sessions` | `Optional[int]` | `None` |
| `target_session_count` | `Optional[int]` | `None` |

---

### Class `PlacementStrategy` *(ABC)*

Abstract base class for placement strategies.

#### Properties

```python
@property
@abstractmethod
def name(self) -> str
```
Strategy name for logging/tracking.

#### Methods

```python
@abstractmethod
def place(
    self,
    embeddings: np.ndarray,
    abstract_ids: List[str],
    constraints: SessionConstraints,
    hybrid_assignments: Optional[Dict[str, str]] = None,
) -> PlacementResult
```
Assign presentations to sessions.

---

### Class `OralSessionPlacement(PlacementStrategy)`

Hierarchical clustering-based placement for oral sessions (No Hybrid).

#### Constructor

```python
def __init__(
    self,
    linkage_method: str = "average",
    tree_merge_stop: float = 0.95,
    similarity_func=None,
)
```
Initialize placement strategy.

#### Properties

```python
@property
def name(self) -> str   # returns "oral_session_hierarchical"
```

#### Methods

```python
def place(
    self,
    embeddings: np.ndarray,
    abstract_ids: List[str],
    constraints: SessionConstraints,
    hybrid_assignments: Optional[Dict[str, str]] = None,
) -> PlacementResult
```
Assign presentations to sessions using bottom-up hierarchical clustering.

---

### Class `HybridFirstPlacement(PlacementStrategy)`

Hierarchical clustering-based placement that fills hybrid sessions first.

#### Constructor

```python
def __init__(
    self,
    linkage_method: str = "average",
    tree_merge_stop: float = 0.95,
    similarity_func=None,
)
```
Initialize placement strategy.

#### Properties

```python
@property
def name(self) -> str   # returns "hybrid_first_hierarchical"
```

#### Methods

```python
def place(
    self,
    embeddings: np.ndarray,
    abstract_ids: List[str],
    constraints: SessionConstraints,
    hybrid_assignments: Optional[Dict[str, str]] = None,
    hybrid_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> PlacementResult
```
Assign presentations to sessions, filling hybrid sessions first.

---

### Class `TraditionalClusterPlacement(PlacementStrategy)`

Traditional fcluster-based placement for comparison/research purposes.

#### Constructor

```python
def __init__(
    self,
    linkage_method: str = "average",
    tree_merge_stop: float = 0.95,
    similarity_func=None,
)
```

#### Properties

```python
@property
def name(self) -> str   # returns "traditional_fcluster"
```

#### Methods

```python
def place(
    self,
    embeddings: np.ndarray,
    abstract_ids: List[str],
    constraints: SessionConstraints,
    hybrid_assignments: Optional[Dict[str, str]] = None,
) -> PlacementResult
```
Assign presentations using traditional fcluster approach.

---

### Class `PosterThematicOrdering(PlacementStrategy)`

Order poster presentations by thematic similarity.

#### Constructor

```python
def __init__(
    self,
    similarity_func=None,
    linkage_method: str = "average",
    tree_merge_stop: float = 0.95,
)
```

#### Properties

```python
@property
def name(self) -> str   # returns "poster_thematic_ordering"
```

#### Methods

```python
def place(
    self,
    embeddings: np.ndarray,
    abstract_ids: List[str],
    constraints: SessionConstraints,
    hybrid_assignments: Optional[Dict[str, str]] = None,
) -> PlacementResult
```
Create thematic ordering using traveling salesman approximation.

---

### Module-Level Functions

```python
def create_placement_strategy(
    strategy_type: str = "oral",
    **kwargs
) -> PlacementStrategy
```
Factory function to create placement strategies.

---

## `smart.core.metrics`

> Metrics calculations for session organization.

### Functions

```python
def calculate_session_coherence(
    embeddings: np.ndarray,
    session_indices: List[int],
) -> float
```
Calculate coherence score for a session.

```python
def calculate_presentation_fit(
    embedding: np.ndarray,
    session_embeddings: np.ndarray,
    exclude_self: bool = True,
) -> float
```
Calculate how well a presentation fits its session.

```python
def calculate_session_distinctiveness(
    session_embeddings: np.ndarray,
    other_session_embeddings: List[np.ndarray],
) -> float
```
Calculate how distinct a session is from other sessions.

```python
def calculate_session_session_similarity(
    session_embeddings_list: List[np.ndarray],
    session_ids: List[str],
) -> Dict[Tuple[str, str], float]
```
Calculate pairwise similarity between all sessions.

```python
def calculate_all_metrics(
    embeddings: np.ndarray,
    abstract_ids: List[str],
    session_assignments: Dict[str, str],
) -> Dict[str, Any]
```
Calculate all metrics for a complete session organization.

```python
def find_outlier_presentations(
    embeddings: np.ndarray,
    abstract_ids: List[str],
    session_assignments: Dict[str, str],
    fit_threshold: float = 0.5,
    std_threshold: float = 2.0,
) -> List[Dict[str, Any]]
```
Find presentations that are poor fits for their sessions.

```python
def find_similar_presentations(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    abstract_ids: List[str],
    top_k: int = 10,
    exclude_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]
```
Find presentations most similar to a query.

---

## `smart.io.loaders`

> Data loaders for SMART system.

---

### `@dataclass` ColumnMapping

Mapping configuration for loading data files.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `title` | `str` | *(required)* |
| `abstract` | `Optional[str]` | `None` |
| `submission_id` | `Optional[str]` | `None` |
| `presenter_first_name` | `Optional[str]` | `None` |
| `presenter_last_name` | `Optional[str]` | `None` |
| `presenter_email` | `Optional[str]` | `None` |
| `affiliation` | `Optional[str]` | `None` |
| `technical_community` | `Optional[str]` | `None` |
| `session_preference` | `Optional[str]` | `None` |
| `session` | `Optional[str]` | `None` |
| `extra_columns` | `Dict[str, str]` | `field(default_factory=dict)` |

#### Methods

```python
def to_rename_dict(self) -> Dict[str, str]
```
Generate pandas rename dictionary.

---

### Class `ColumnMapper`

Interactive column mapper for data files.

#### Constants

| Name | Type | Description |
|------|------|-------------|
| `PATTERNS` | `Dict[str, List[str]]` | Regex patterns for auto-detecting common column names |

#### Constructor

```python
def __init__(self, df: pd.DataFrame)
```
Initialize mapper with source dataframe.

#### Methods

```python
def get_suggestions(self) -> Dict[str, Optional[str]]
```
Get auto-detected mapping suggestions.

```python
def get_unmapped_columns(self) -> List[str]
```
Get columns not yet mapped to standard fields.

```python
def preview_column(self, column: str, n: int = 5) -> List[Any]
```
Preview values from a column.

```python
def create_mapping(
    self,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
    submission_id: Optional[str] = None,
    **kwargs
) -> ColumnMapping
```
Create column mapping, using auto-detected values as defaults.

---

### Module-Level Functions

```python
def read_file(file_path: str | Path) -> pd.DataFrame
```
Read CSV or Excel file into DataFrame.

```python
def generate_abstract_id(
    submission_id: Optional[str],
    conference_year: int,
    temp_counter: int
) -> Tuple[str, bool]
```
Generate abstract ID from submission ID or create temporary ID.

```python
def load_presentations(
    file_path: str | Path,
    mapping: ColumnMapping,
    conference_year: int = 26,
    drop_missing_title: bool = True,
    drop_missing_abstract: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]
```
Load presentations from file with column mapping.

```python
def load_hybrid_sessions(
    file_path: str | Path,
    mapping: ColumnMapping,
    conference_year: int = 26,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
```
Load hybrid sessions with pre-assigned presentations.

```python
def load_committees(
    file_path: str | Path,
    name_column: str = "Committee_Name",
    description_column: str = "Description",
) -> Tuple[pd.DataFrame, Dict[str, Any]]
```
Load committee data for session assignment.

```python
def inspect_file(file_path: str | Path) -> Dict[str, Any]
```
Inspect a data file and return information about its structure.

---

## `smart.io.exporters`

> Export functionality for SMART system.

---

### Enum `ExportFormat`

Supported export formats.

| Member | Value |
|--------|-------|
| `SQLITE` | `"sqlite"` |
| `PARQUET` | `"parquet"` |
| `CSV` | `"csv"` |
| `EXCEL` | `"excel"` |

---

### `@dataclass` ExportProfile

Configuration for export operations.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `name` | `str` | *(required)* |
| `description` | `str` | *(required)* |
| `include_fields` | `Optional[List[str]]` | `None` |
| `redact_fields` | `List[str]` | `field(default_factory=list)` |
| `anonymize_fields` | `List[str]` | `field(default_factory=list)` |
| `include_embeddings` | `bool` | `False` |
| `include_similarities` | `bool` | `True` |
| `include_history` | `bool` | `False` |
| `format` | `ExportFormat` | `ExportFormat.PARQUET` |

---

### `@dataclass` ViewerBundle

Container for viewer data bundle.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `presentations` | `pd.DataFrame` | *(required)* |
| `sessions` | `pd.DataFrame` | *(required)* |
| `pres_similarities` | `pd.DataFrame` | *(required)* |
| `session_similarities` | `pd.DataFrame` | *(required)* |
| `embeddings` | `Optional[np.ndarray]` | `None` |
| `abstract_ids` | `Optional[List[str]]` | `None` |
| `manifest` | `Dict[str, Any]` | `field(default_factory=dict)` |

#### Methods

```python
def validate(self) -> List[str]
```
Validate bundle integrity.

---

### Predefined Export Profiles

| Constant | Profile Name | Description |
|----------|-------------|-------------|
| `PROFILE_CLOUD_VIEWER` | `"cloud_viewer"` | Export for public cloud viewer — no PII, no abstracts by default |
| `PROFILE_ORGANIZER_FULL` | `"organizer_full"` | Full export for organizers — includes all data |
| `PROFILE_ROOM_ASSIGNMENT` | `"room_assignment"` | Spreadsheet for room and time assignment |
| `PROFILE_PRESENTER_LIST` | `"presenter_list"` | Contact list for presenter communication |

---

### Module-Level Functions

```python
def get_profile(name: str) -> ExportProfile
```
Get a predefined export profile by name.

```python
def export_presentations(
    conference_db,
    profile: ExportProfile,
    output_path: str | Path,
) -> Dict[str, Any]
```
Export presentations according to profile.

```python
def export_sessions(
    conference_db,
    profile: ExportProfile,
    output_path: str | Path,
) -> Dict[str, Any]
```
Export sessions according to profile.

```python
def export_similarity_matrix(
    embeddings: np.ndarray,
    ids: List[str],
    output_path: str | Path,
    similarity_func=None,
) -> Dict[str, Any]
```
Export pre-computed similarity matrix.

```python
def export_for_viewer(
    conference_db,
    embeddings: np.ndarray,
    abstract_ids: List[str],
    output_dir: str | Path,
    profile: Optional[ExportProfile] = None,
    version_tag: Optional[str] = None,
) -> Dict[str, Any]
```
Export complete package for cloud viewer deployment.

```python
def export_for_organizers(
    conference_db,
    embedding_cache,
    output_path: str | Path,
    include_cache: bool = True,
) -> Dict[str, Any]
```
Export complete SQLite database for organizers.

```python
def export_spreadsheet(
    conference_db,
    output_path: str | Path,
    include_sessions: bool = True,
    include_presentations: bool = True,
    profile: Optional[ExportProfile] = None,
) -> Dict[str, Any]
```
Export to Excel spreadsheet with multiple sheets.

```python
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
) -> Dict[str, Any]
```
Export a complete viewer bundle as a directory or zip file.

```python
def load_viewer_bundle(bundle_path: str | Path) -> ViewerBundle
```
Load a viewer bundle from directory or zip file.

---

## `smart.llm.embeddings`

> Embedding backends for SMART system.

---

### Class `EmbeddingBackend` *(ABC)*

Abstract base class for embedding backends.

#### Properties

```python
@property
@abstractmethod
def model_name(self) -> str
```
Return the model name.

```python
@property
@abstractmethod
def model_version(self) -> str
```
Return the model version string.

#### Methods

```python
@abstractmethod
def embed(self, text: str) -> np.ndarray
```
Generate embedding for a single text.

```python
@abstractmethod
def embed_batch(self, texts: List[str]) -> List[np.ndarray]
```
Generate embeddings for multiple texts.

```python
def get_config(self, task_type: str = "SEMANTIC_SIMILARITY") -> EmbeddingConfig
```
Get embedding configuration for this backend.

---

### Class `CachedEmbedder`

Wrapper that adds caching to any embedding backend.

#### Constructor

```python
def __init__(
    self,
    backend: EmbeddingBackend,
    cache: EmbeddingCache,
    task_type: str = "SEMANTIC_SIMILARITY"
)
```

#### Properties

```python
@property
def config(self) -> EmbeddingConfig
```

```python
@property
def truncation_count(self) -> int
```
Get truncation count from backend if supported.

```python
@property
def truncated_ids(self) -> List[str]
```
Get truncated IDs from backend if supported.

#### Methods

```python
def reset_truncation_stats(self)
```
Reset truncation stats on backend if supported.

```python
def embed(self, text: str, text_id: Optional[str] = None) -> np.ndarray
```
Get embedding, using cache if available.

```python
def embed_batch(
    self,
    texts: List[str],
    text_ids: Optional[List[str]] = None,
    show_progress: bool = False
) -> List[np.ndarray]
```
Get embeddings for multiple texts, using cache where available.

```python
def get_stats(self) -> Dict[str, Any]
```
Get combined stats from backend and cache.

---

### Class `GeminiEmbedder(EmbeddingBackend)`

Embedding backend using Google Gemini API.

#### Constants

| Name | Value |
|------|-------|
| `DEFAULT_MODEL` | `"gemini-embedding-001"` |

#### Constructor

```python
def __init__(
    self,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    task_type: str = "SEMANTIC_SIMILARITY",
    delay_seconds: float = 0.0
)
```
Initialize Gemini embedder.

#### Properties

```python
@property
def model_name(self) -> str
```

```python
@property
def model_version(self) -> str
```

#### Methods

```python
def embed(self, text: str) -> np.ndarray
```
Generate embedding for single text.

```python
def embed_batch(self, texts: List[str]) -> List[np.ndarray]
```
Generate embeddings for multiple texts.

---

### Class `OllamaEmbedder(EmbeddingBackend)`

Embedding backend using local Ollama server.

#### Constants

| Name | Value |
|------|-------|
| `DEFAULT_MODEL` | `"nomic-embed-text-v2-moe"` |
| `DEFAULT_HOST` | `"http://localhost:11434"` |
| `DEFAULT_MAX_CHARS` | `8000` |

#### Constructor

```python
def __init__(
    self,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 120.0,
    max_chars: Optional[int] = None
)
```
Initialize Ollama embedder.

#### Properties

```python
@property
def model_name(self) -> str
```

```python
@property
def model_version(self) -> str
```

```python
@property
def truncation_count(self) -> int
```
Number of texts that were truncated due to context length limits.

```python
@property
def truncated_ids(self) -> List[str]
```
List of text identifiers that were truncated.

#### Methods

```python
def reset_truncation_stats(self)
```
Reset truncation tracking for a new batch.

```python
def embed(self, text: str, text_id: Optional[str] = None) -> np.ndarray
```
Generate embedding for single text.

```python
def embed_batch(self, texts: List[str], text_ids: Optional[List[Optional[str]]] = None) -> List[np.ndarray]
```
Generate embeddings for multiple texts.

---

### Class `SentenceTransformerEmbedder(EmbeddingBackend)`

Embedding backend using sentence-transformers library.

#### Constants

| Name | Value |
|------|-------|
| `DEFAULT_MODEL` | `"all-MiniLM-L6-v2"` |

#### Constructor

```python
def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None)
```
Initialize sentence-transformers embedder.

#### Properties

```python
@property
def model_name(self) -> str
```

```python
@property
def model_version(self) -> str
```

#### Methods

```python
def embed(self, text: str) -> np.ndarray
```
Generate embedding for single text.

```python
def embed_batch(self, texts: List[str]) -> List[np.ndarray]
```
Generate embeddings for multiple texts.

---

### Module-Level Functions

```python
def get_ollama_models(
    host: str = "http://localhost:11434",
    timeout: float = 10.0,
    filter_embedding: bool = False
) -> List[str]
```
Discover available models from Ollama server.

```python
def check_ollama_connection(host: str = "http://localhost:11434", timeout: float = 5.0) -> bool
```
Check if Ollama server is reachable.

```python
def create_embedder(
    backend: str = "gemini",
    model: Optional[str] = None,
    cache: Optional[EmbeddingCache] = None,
    task_type: str = "SEMANTIC_SIMILARITY",
    **kwargs
) -> EmbeddingBackend | CachedEmbedder
```
Factory function to create an embedder.

---

## `smart.llm.titles`

> Title and keyword generation for sessions.

---

### `@dataclass` TitleGenerationResult

Result from title generation.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `titles` | `List[str]` | *(required)* |
| `keywords` | `List[str]` | *(required)* |
| `model_name` | `str` | *(required)* |
| `raw_response` | `Optional[str]` | `None` |

---

### Class `TitleGenerator` *(ABC)*

Abstract base class for title generation backends.

#### Properties

```python
@property
@abstractmethod
def model_name(self) -> str
```
Return the model name.

#### Methods

```python
@abstractmethod
def generate(
    self,
    presentations: List[Dict[str, str]],
    prompt_template: Optional[str] = None,
) -> TitleGenerationResult
```
Generate titles and keywords for a session.

---

### Class `GeminiTitleGenerator(TitleGenerator)`

Title generator using Google Gemini API.

#### Constants

| Name | Value |
|------|-------|
| `DEFAULT_MODEL` | `"gemini-2.5-flash-lite"` |

#### Constructor

```python
def __init__(
    self,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
)
```
Initialize Gemini title generator.

#### Properties

```python
@property
def model_name(self) -> str
```

#### Methods

```python
def generate(
    self,
    presentations: List[Dict[str, str]],
    prompt_template: Optional[str] = None,
) -> TitleGenerationResult
```
Generate titles using Gemini.

---

### Class `OllamaTitleGenerator(TitleGenerator)`

Title generator using local Ollama server.

#### Constants

| Name | Value |
|------|-------|
| `DEFAULT_MODEL` | `"llama3.2"` |
| `DEFAULT_HOST` | `"http://localhost:11434"` |

#### Constructor

```python
def __init__(
    self,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 120.0,
)
```
Initialize Ollama title generator.

#### Properties

```python
@property
def model_name(self) -> str
```

#### Methods

```python
def generate(
    self,
    presentations: List[Dict[str, str]],
    prompt_template: Optional[str] = None,
) -> TitleGenerationResult
```
Generate titles using Ollama.

---

### Class `CachedTitleGenerator`

Wrapper that caches generated titles by presentation set hash.

#### Constructor

```python
def __init__(self, generator: TitleGenerator, conference_db)
```
Initialize cached generator.

#### Methods

```python
def generate(
    self,
    session_id: str,
    presentations: List[Dict[str, str]],
    force_regenerate: bool = False,
    prompt_template: Optional[str] = None,
) -> TitleGenerationResult
```
Generate titles, using cache if available.

---

### Module-Level Constants

| Name | Type | Description |
|------|------|-------------|
| `DEFAULT_PROMPT_TEMPLATE` | `str` | Default prompt for session title/keyword generation |

### Module-Level Functions

```python
def get_ollama_models(
    host: str = "http://localhost:11434",
    timeout: float = 10.0,
    filter_generation: bool = False
) -> List[str]
```
Discover available models from Ollama server.

```python
def check_ollama_connection(host: str = "http://localhost:11434", timeout: float = 5.0) -> bool
```
Check if Ollama server is reachable.

```python
def generate_session_hash(presentation_ids: List[str]) -> str
```
Generate hash for a set of presentations.

```python
def create_title_generator(
    backend: str = "gemini",
    model: Optional[str] = None,
    **kwargs
) -> TitleGenerator
```
Factory function to create title generators.

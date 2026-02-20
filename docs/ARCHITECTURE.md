# SMART Architecture

**Last Updated**: February 2026  
**Purpose**: Living technical reference documenting the current system state. Update this document when code changes.

---

## System Overview

SMART (Session Matching And Automated Recommendation Tool) automates conference session organization using text embeddings and clustering algorithms. The system has two main components:

1. **SMART Library** (`smart/`) - Core algorithms and data management
2. **Applications** - User interfaces that consume the library

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SMART ECOSYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SMART Library (smart/)                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ database │  │embeddings│  │ placement│  │ exporters│            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│          ┌─────────────────────────┼─────────────────────────┐             │
│          ▼                         ▼                         ▼             │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐       │
│  │  smart_app   │         │ progress_    │         │   Viewer     │       │
│  │  (Wizard)    │         │ tracker      │         │   App        │       │
│  │              │         │              │         │              │       │
│  │ Phase 1-3:   │         │ Phase 4:     │         │ Phase 2+:    │       │
│  │ Initial      │         │ Ongoing      │         │ Exploration  │       │
│  │ Creation     │         │ Updates      │         │              │       │
│  └──────────────┘         └──────────────┘         └──────────────┘       │
│          │                         │                         ▲             │
│          │                         │                         │             │
│          └─────────────────────────┴─────────────────────────┘             │
│                                    │                                        │
│                            Export Bundle                                    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Standalone Deployed Viewers                      │   │
│  │         (Viewer app + export bundle → separate repository)          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Project Directory Structure

```
Session-Creation-Package/
├── smart/                  # Core library
│   ├── core/               # Database, placement, metrics
│   ├── io/                 # Loaders, exporters
│   └── llm/                # Embeddings, title generation
├── apps/                   # Streamlit applications
│   ├── smart_app.py        # Session creation wizard
│   ├── session_progress_tracker.py
│   └── session_viewer_app.py
├── scripts/                # CLI tools
│   ├── smart_cli.py        # Batch processing
│   └── migrate_data.py     # Data migration
├── notebooks/              # Current Jupyter notebooks
├── data/                   # Input data files
│   ├── submissions/        # Presentation CSVs
│   ├── committees/         # Committee assignments
│   └── hybrid/             # Hybrid session data
├── databases/              # SQLite databases (*_cache.db, *_working.db)
├── docs/                   # Documentation
├── tests/                  # Test suite
└── legacy/                 # Deprecated files
    ├── notebooks/          # Old session_organizer notebooks
    └── *.py                # Old Python files
```

---

## SMART Library (`smart/`)

The core library providing all session organization functionality.

### Module Structure

```
smart/
├── __init__.py              # Package exports
├── core/
│   ├── __init__.py
│   ├── database.py          # SQLite persistence
│   ├── placement.py         # Clustering algorithms
│   └── metrics.py           # Quality evaluation
├── io/
│   ├── __init__.py
│   ├── loaders.py           # Data import
│   └── exporters.py         # Data export
└── llm/
    ├── __init__.py
    ├── embeddings.py        # Embedding backends
    └── titles.py            # Title generation
```

### Core Modules

#### `smart/core/database.py`

**Purpose**: SQLite-based persistence for embeddings and conference data.

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `EmbeddingConfig` | Identifies embedding model configuration (name, version, task_type, dimensions) |
| `EmbeddingCache` | Stores/retrieves embeddings keyed by text hash + config hash |
| `ConferenceDB` | Stores presentations, sessions, committees, placements, titles |

**Database Files**:
- `{conference}_cache.db` - Embedding cache (reusable across workflows)
- `{conference}_working.db` - Working data (presentations, sessions, etc.)

**EmbeddingCache Methods**:

| Method | Purpose |
|--------|---------|
| `get_embedding(text, config)` | Get single cached embedding |
| `get_embeddings_batch(texts, config)` | Get multiple cached embeddings |
| `store_embedding(text, embedding, config)` | Store single embedding |
| `store_embeddings_batch(items, config)` | Store multiple embeddings |
| `get_coverage_by_model(texts)` | Get cache coverage per model for given texts |
| `get_all_model_configs()` | List all model configurations in cache |
| `get_stats()` | Get cache statistics |

**EmbeddingCache Schema**:
```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    text_hash TEXT NOT NULL,           -- SHA-256 of input text
    config_hash TEXT NOT NULL,         -- Hash of model config
    model_name TEXT NOT NULL,
    model_version TEXT,
    task_type TEXT,
    dimensions INTEGER,
    embedding BLOB NOT NULL,           -- numpy array as bytes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(text_hash, config_hash)
);

CREATE TABLE model_registry (
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    task_type TEXT NOT NULL,
    dimensions INTEGER,
    first_used TIMESTAMP,
    last_used TIMESTAMP,
    embedding_count INTEGER,
    PRIMARY KEY (model_name, model_version, task_type)
);
```

**ConferenceDB Tables**:
- `presentations` - Title, abstract, submission ID, presenter info
- `sessions` - Session ID, title, keywords, metrics
- `placements` - Presentation-to-session assignments
- `committees` - Committee names and descriptions
- `titles` - Generated title options per session

#### `smart/core/placement.py`

**Purpose**: Session assignment algorithms.

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `SessionConstraints` | Min/max session size, max sessions |
| `PlacementStrategy` | Abstract base for placement algorithms |
| `HybridFirstStrategy` | Place hybrid sessions first, then cluster remaining |
| `PlacementResult` | Sessions and assignments output |

**Algorithm**: Agglomerative clustering with ward linkage, respecting size constraints.

#### `smart/core/metrics.py`

**Purpose**: Session quality evaluation.

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `calculate_all_metrics()` | Compute coherence, distinctiveness for all sessions |
| `find_outlier_presentations()` | Identify presentations that don't fit their session |
| `calculate_session_coherence()` | Average pairwise similarity within session |
| `calculate_session_distinctiveness()` | How different session is from others |

### I/O Modules

#### `smart/io/loaders.py`

**Purpose**: Flexible data import from CSV/Excel.

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `ColumnMapper` | Maps source columns to target fields |
| `ColumnMapping` | Defines a single column mapping |

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `inspect_file()` | Preview file columns and sample data |
| `load_presentations()` | Import presentations with column mapping |
| `load_hybrid_sessions()` | Import pre-assigned hybrid session data |
| `load_committees()` | Import committee information |

#### `smart/io/exporters.py`

**Purpose**: Export data for various consumers.

**Export Profiles**:

| Profile | Purpose | Content |
|---------|---------|---------|
| `PROFILE_CLOUD_VIEWER` | Public web viewer | Redacted PII, no abstracts |
| `PROFILE_ORGANIZER_FULL` | Organizer review | Full data with embeddings |
| `PROFILE_ROOM_ASSIGNMENT` | Room scheduling | Session list with sizes |
| `PROFILE_PRESENTER_LIST` | Contact list | Presenter details |

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `export_viewer_bundle()` | Create bundle directory with all viewer data |
| `export_spreadsheet()` | Export to Excel/CSV |

**Viewer Bundle Contents**:
```
{conference}_bundle/
├── manifest.json              # Bundle metadata
├── presentations.parquet      # Presentation data
├── presentations_public.parquet  # Redacted version
├── sessions.parquet           # Session data
├── pres_similarities.parquet  # Presentation similarity matrix
└── session_similarities.parquet  # Session similarity matrix (optional)
```

### LLM Modules

#### `smart/llm/embeddings.py`

**Purpose**: Text embedding generation.

**Embedding Backends**:

| Class | Provider | Model Default |
|-------|----------|---------------|
| `GeminiEmbedder` | Google AI | `gemini-embedding-001` |
| `OllamaEmbedder` | Ollama (local) | `nomic-embed-text-v2-moe` |
| `SentenceTransformerEmbedder` | HuggingFace | `all-MiniLM-L6-v2` |
| `CachedEmbedder` | Wrapper | Caches any backend |

**Key Functions**:

| Function | Purpose |
|----------|---------|
| `create_embedder()` | Factory for embedding backends |
| `get_ollama_models()` | List available Ollama models |
| `check_ollama_connection()` | Verify Ollama server is running |

#### `smart/llm/titles.py`

**Purpose**: LLM-based title and keyword generation.

**Title Generators**:

| Class | Provider | Model Default |
|-------|----------|---------------|
| `GeminiTitleGenerator` | Google AI | `gemini-2.5-flash-lite` |
| `OllamaTitleGenerator` | Ollama (local) | `llama3.2` |
| `CachedTitleGenerator` | Wrapper | Caches any generator |

---

## Applications

### Directory: `apps/`

All Streamlit applications are in the `apps/` directory.

#### `apps/smart_app.py` - Session Creation Wizard

**Purpose**: Full workflow for initial session creation.

**Phase**: 1-3 (Initial Session Creation)

**Workflow Steps**:
1. Conference Setup - Name, year, database paths
2. Import Data - Presentations, hybrid sessions, committees
3. Generate Embeddings - Select model, generate/load from cache
4. Create Sessions - Clustering with size constraints
5. Generate Titles - LLM title and keyword generation
6. Export - Bundle for viewer, spreadsheets for portal

**Launch**: `streamlit run apps/smart_app.py`

#### `apps/session_progress_tracker.py` - Progress Tracking

**Purpose**: Track session organization through the year when sessions already exist.

**Phase**: 4 (Iterative Updates)

**Use Case**: Sessions have been created externally (in portal). This app:
- Imports current session assignments
- Generates/loads embeddings
- Calculates metrics for existing sessions
- Exports viewer bundles for team review

**Launch**: `streamlit run apps/session_progress_tracker.py`

#### `apps/session_viewer_app.py` - Session Viewer

**Purpose**: Interactive exploration of sessions and presentations.

**Phase**: 2+ (Ongoing Organization & Attendee Discovery)

**Features**:
- Browse sessions with coherence/distinctiveness metrics
- View presentation details
- Find similar presentations
- Search and filter

**Input**: Viewer bundle (directory or zip)

**Launch**: `streamlit run apps/session_viewer_app.py`

### Deployment Pattern: Standalone Viewers

The viewer app is designed to be deployed as standalone websites:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VIEWER DEPLOYMENT WORKFLOW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Create export bundle from smart_app or session_progress_tracker        │
│                                                                             │
│     output/AIM2026_bundle/                                                  │
│     ├── manifest.json                                                       │
│     ├── presentations.parquet                                               │
│     ├── sessions.parquet                                                    │
│     └── pres_similarities.parquet                                           │
│                                                                             │
│  2. Create new repository for deployment                                    │
│                                                                             │
│     aim2026-viewer/                                                         │
│     ├── session_viewer_app.py    ← Copy from apps/session_viewer_app.py    │
│     ├── data/                    ← Export bundle contents                   │
│     │   ├── manifest.json                                                   │
│     │   └── ...                                                             │
│     ├── requirements.txt                                                    │
│     └── .streamlit/                                                         │
│         └── secrets.toml         ← Access password if needed                │
│                                                                             │
│  3. Deploy to Streamlit Cloud                                               │
│                                                                             │
│     → https://aim2026-viewer.streamlit.app/                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why Copy Instead of Submodule?**
- Simpler deployment (single repository)
- Can customize viewer per conference
- No dependency on main repository
- Viewers remain stable even if main repo changes

**Year-Specific Viewer Files**:

Previous deployment bases are now in `legacy/` for reference:
- `legacy/session_creation_viewer_web_app.py` - AIM 2025 base
- `legacy/session_creation_viewer_web_app26.py` - AIM 2026 base

The canonical viewer for new deployments is `apps/session_viewer_app.py`.

### CLI Tools (Directory: `scripts/`)

#### `scripts/smart_cli.py`

**Purpose**: Command-line interface for batch processing.

**Usage**:
```bash
python scripts/smart_cli.py \
    --presentations abstracts.xlsx \
    --hybrid hybrid.xlsx \
    --output output/ \
    --conference AIM2026
```

#### `scripts/migrate_data.py`

**Purpose**: Migrate existing parquet/CSV data into SQLite databases.

**Usage**:
```bash
python scripts/migrate_data.py \
    --source presentations_with_embeddings.parquet \
    --target AIM2026
```

---

## Legacy System (Directory: `legacy/`)

### Overview

The original monolithic implementation before the SMART library refactor.

**Status**: DEPRECATED - Functionality moved to `smart/` library.

**File**: `legacy/session_organizer.py` (1728 lines)

### Dependent Files

These files import `session_organizer` and are part of the legacy system:

| File | Type | Purpose |
|------|------|---------|
| `legacy/session_creation_app_v2.py` | Tkinter GUI | Desktop application |
| `legacy/notebooks/session creation operation.ipynb` | Notebook | Original workflow documentation |
| `legacy/notebooks/session creation operation 2026_old.ipynb` | Notebook | 2026-specific workflow |
| `legacy/notebooks/session creation operation 2026_simple.ipynb` | Notebook | Simplified workflow |
| `legacy/notebooks/SMART 2026 Initial Sort.ipynb` | Notebook | Initial sort for AIM 2026 |
| `legacy/notebooks/Processing Steps.ipynb` | Notebook | Processing documentation |

### Key Differences from SMART Library

| Aspect | Legacy (`session_organizer`) | Current (`smart/`) |
|--------|------------------------------|---------------------|
| Persistence | Parquet/pickle files | SQLite databases |
| Caching | Manual file management | Automatic by text+config hash |
| Embeddings | Sentence-Transformers only | Gemini, Ollama, ST |
| Export | Manual DataFrame manipulation | Export profiles |
| Structure | Single 1700-line file | Modular packages |

---

## Data Flow

### Initial Session Creation Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ CSV/Excel   │───▶│   Loaders   │───▶│ ConferenceDB│
│ Input Files │    │             │    │ (SQLite)    │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Embedding   │◀───│  Embedders  │◀───│ Presentations│
│ Cache (SQL) │    │ (Gemini/    │    │ text        │
│             │    │  Ollama)    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Embedding   │───▶│  Placement  │───▶│  Sessions   │
│ Vectors     │    │ (Clustering)│    │ + Metrics   │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Viewer     │◀───│  Exporters  │◀───│   Titles    │
│  Bundle     │    │             │    │ (LLM)       │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Progress Tracking Flow (Phase 4)

```
┌─────────────┐    ┌─────────────┐
│ Portal Data │───▶│   Import    │  Sessions already assigned
│ (CSV/Excel) │    │             │  externally in portal
└─────────────┘    └─────────────┘
                         │
                         ▼
┌─────────────┐    ┌─────────────┐
│ Embedding   │◀──▶│  Generate/  │  Use cache if available
│ Cache       │    │  Load       │
└─────────────┘    └─────────────┘
                         │
                         ▼
┌─────────────┐    ┌─────────────┐
│  Metrics    │◀───│  Calculate  │  Coherence, distinctiveness
│             │    │             │
└─────────────┘    └─────────────┘
                         │
                         ▼
┌─────────────┐    ┌─────────────┐
│  Viewer     │◀───│   Export    │  For team review
│  Bundle     │    │             │
└─────────────┘    └─────────────┘
```

---

## File Status Reference

See [FILE_INVENTORY.md](FILE_INVENTORY.md) for complete file listing with status indicators.

### Quick Reference

| Category | Current (SMART) | Legacy | Deployment Artifacts |
|----------|-----------------|--------|----------------------|
| **Main Apps** | `smart_app.py`, `session_viewer_app.py`, `session_progress_tracker.py` | `session_creation_app_v2.py` | - |
| **CLI** | `smart_cli.py`, `migrate_data.py` | - | - |
| **Library** | `smart/` (8 modules) | `session_organizer.py` | - |
| **Notebooks** | `SMART_Demo.ipynb` | 5+ notebooks | - |
| **Viewers** | `session_viewer_app.py` | - | `session_creation_viewer_web_app*.py` |

---

## Configuration

### Environment Variables (`.env`)

```env
# Required for Gemini
GOOGLE_API_KEY=your_api_key

# Required for encrypted exports
DATAFRAME_PW=encryption_password

# Optional: Ollama server URL (default: http://localhost:11434)
OLLAMA_HOST=http://localhost:11434
```

### Streamlit Secrets (`.streamlit/secrets.toml`)

For deployed viewers with password protection:

```toml
access_password = "your_viewer_password"
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-01 | SMART library refactor, SQLite persistence |
| 1.x | 2025 | Original `session_organizer.py` implementation |

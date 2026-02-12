# SMART Project - File Inventory

**Generated**: February 2026  
**Last Updated**: February 12, 2026 (after reorganization)  
**Purpose**: Document all files in the project with their status and purpose.

---

## Status Legend

| Status | Description |
|--------|-------------|
| ✅ **CURRENT** | Active, maintained code using SMART library |
| 🔶 **TRANSITIONAL** | Working code that mixes legacy and current approaches |
| ❌ **LEGACY** | Depends on `session_organizer.py` - to be deprecated |
| 📦 **DATA** | Data files, not code |
| 📝 **DOCS** | Documentation |
| 🗑️ **CANDIDATE FOR DELETION** | No longer needed |

---

## Directory Structure (Post-Reorganization)

```
Session-Creation-Package/
├── smart/              # Core library
├── apps/               # Streamlit applications
├── scripts/            # CLI tools
├── notebooks/          # Current notebooks
├── data/               # Input data files
│   ├── submissions/
│   ├── committees/
│   └── hybrid/
├── databases/          # SQLite databases
├── output/             # Export outputs
├── exports/            # Spreadsheet exports
├── tests/              # Test suite (planned)
├── docs/               # Documentation
└── legacy/             # Deprecated files
    └── notebooks/
```

---

## Applications (Streamlit)

### ✅ CURRENT - Using SMART Library

| File | Lines | Purpose | Phase |
|------|-------|---------|-------|
| `apps/smart_app.py` | 2410 | Main wizard for session creation | Phase 1-3 |
| `apps/session_viewer_app.py` | 492 | Viewer for exploring sessions | Phase 2 |
| `apps/session_progress_tracker.py` | 662 | Track progress for pre-assigned sessions | Phase 4 |

**Launch Commands**:
```bash
streamlit run apps/smart_app.py
streamlit run apps/session_viewer_app.py
streamlit run apps/session_progress_tracker.py
```

### ❌ LEGACY - Moved to `legacy/`

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `legacy/session_creation_app_v2.py` | 3065 | Tkinter GUI desktop app | **DEPRECATED** |
| `legacy/session_creation_viewer_web_app.py` | 366 | AIM 2025 specific viewer | **DEPRECATED** |
| `legacy/session_creation_viewer_web_app26.py` | 365 | AIM 2026 specific viewer | **DEPRECATED** |

---

## Command Line Tools

### ✅ CURRENT - Moved to `scripts/`

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/smart_cli.py` | 633 | CLI for batch processing |
| `scripts/migrate_data.py` | 423 | Migration script for legacy data |

### ❌ LEGACY - Moved to `legacy/`

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `session_organizer.py` | 1728 | Original monolithic algorithm engine | **DEPRECATED** - Functions moved to `smart/` library |

---

## SMART Library (`smart/`)

### ✅ CURRENT - Core Package

| File | Purpose |
|------|---------|
| `smart/__init__.py` | Package exports |
| `smart/core/database.py` | SQLite databases (EmbeddingCache, ConferenceDB) |
| `smart/core/placement.py` | Session assignment algorithms (agglomerative clustering) |
| `smart/core/metrics.py` | Coherence, distinctiveness, outlier detection |
| `smart/io/loaders.py` | CSV/Excel import with flexible column mapping |
| `smart/io/exporters.py` | Export profiles (cloud viewer, organizer, room assignment) |
| `smart/llm/embeddings.py` | Embedding backends (Gemini, Ollama, SentenceTransformers) |
| `smart/llm/titles.py` | LLM title/keyword generation (Gemini, Ollama) |

---

## Jupyter Notebooks

### ✅ CURRENT - Uses SMART Library

| Notebook | Cells | Purpose |
|----------|-------|---------|
| `SMART_Demo.ipynb` | 41 | Complete workflow demonstration using SMART library |

### ❌ LEGACY - Uses `session_organizer.py`

| Notebook | Cells | Purpose | Status |
|----------|-------|---------|--------|
| `session creation operation.ipynb` | 120 | Original comprehensive workflow notebook | **DEPRECATED** - Replace with notebooks using SMART |
| `session creation operation 2026_old.ipynb` | 121 | 2026-specific version | **DEPRECATED** |
| `session creation operation 2026_simple.ipynb` | ? | Simplified version | **DEPRECATED** |
| `SMART 2026 Initial Sort.ipynb` | ? | Initial sort for AIM 2026 | Uses `session_organizer` - evaluate |
| `Processing Steps.ipynb` | ? | Processing documentation | Uses `session_organizer` - evaluate |
| `Data Export.ipynb` | ? | Export workflows | Evaluate dependencies |

---

## Documentation Files

### 📝 Project Documentation

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Project overview | ⚠️ **NEEDS UPDATE** - Still references legacy files as primary |
| `docs/ROADMAP.md` | High-level phases and milestones | Current |
| `docs/BACKLOG.md` | Task tracking | Current |
| `docs/TESTING.md` | Test strategy | Current |

### 📝 Research Papers

| File | Purpose |
|------|---------|
| `Optimizing ASABE AIM Session Creation with Text Clustering and LLMs R1.txt` | Research paper describing algorithm |
| `Text Embedding for Session Organization_R1.txt` | Research paper on embedding approach |

### 📝 Other

| File | Purpose |
|------|---------|
| `Automated Technical Session Creation Flyer.html` | Project flyer/overview |

---

## Data Files

### 📦 Input Data (Examples)

| File | Purpose |
|------|---------|
| `Submissions_852925_all.csv` | Raw submission data |
| `Submissions_852925_completed (1).csv` | Processed submissions |
| `Submissions_852925_completed Final.csv` | Final processed submissions |
| `ASABE Committees.csv` | Committee information |
| `ASABE Committees No ISO TAG.csv` | Committees without ISO TAG |
| `Hybrid Session Invited Presentations.xlsx` | Hybrid session assignments |
| `1.29.25 Abstracts.xlsx` / `abstracts 1.20.26.xlsx` | Abstract data |

### 📦 Working Databases (SQLite)

| Pattern | Purpose |
|---------|---------|
| `*_cache.db` | Embedding cache databases |
| `*_working.db` | Conference working databases (presentations, sessions) |

**Current databases**:
- `AIM26-Test*_cache.db` / `*_working.db` - Various test runs
- `AIM26-MemberHour_*` - Member hour testing
- `AIM26-TOllama1_*` - Ollama testing
- `TAIM26_*` - Additional testing

### 📦 Legacy Data Files (Parquet/PKL)

| File | Purpose | Status |
|------|---------|--------|
| `presentations_with_embeddings.parquet` | Legacy format | 🗑️ Migrate to SQLite |
| `hybrid_presentations_with_embeddings.parquet` | Legacy format | 🗑️ Migrate to SQLite |
| `committees_with_embeddings_backup.parquet` | Legacy backup | 🗑️ Migrate to SQLite |
| `*_backup.parquet` | Various backups | 🗑️ Clean up |
| `embeddings_backup.pkl` | Legacy pickle backup | 🗑️ Clean up |
| `workspace_variables_*.pkl` | Jupyter workspace dumps | 🗑️ Clean up |
| `oral_sessions_*.csv/parquet` | Legacy session outputs | Evaluate |
| `oral_presentations_*.parquet` | Legacy presentation outputs | Evaluate |
| `pres_similarities_matrixAIM26.parquet` | Similarity matrix | Used by legacy viewer |
| `session_similarities_matrixAIM26.parquet` | Session similarity | Used by legacy viewer |
| `df_no_abstractAIM26.parquet` | Presentations without abstracts | Used by legacy viewer |
| `df_sessionsAIM26.parquet` | Session data | Used by legacy viewer |
| `encrypted_dfAIM26.crypt` | Encrypted full data | Used by legacy viewer |

### 📦 Output Directories

| Directory | Purpose |
|-----------|---------|
| `output/` | Export bundles from SMART apps |
| `exports/` | Room assignments, presenter lists |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables (API keys, passwords) |
| `.gitignore` | Git ignore rules |
| `.streamlit/` | Streamlit configuration |
| `requirements.txt` | Python dependencies |

---

## Dependency Map

```
session_organizer.py (LEGACY)
    ↑ imports
    ├── session_creation_app_v2.py (LEGACY GUI)
    ├── session creation operation.ipynb (LEGACY)
    ├── session creation operation 2026_old.ipynb (LEGACY)
    ├── session creation operation 2026_simple.ipynb (LEGACY)
    ├── SMART 2026 Initial Sort.ipynb (LEGACY)
    └── Processing Steps.ipynb (LEGACY)

smart/ (CURRENT LIBRARY)
    ↑ imports
    ├── smart_app.py ✅
    ├── smart_cli.py ✅
    ├── session_viewer_app.py ✅
    ├── session_progress_tracker.py ✅
    ├── migrate_data.py ✅
    └── SMART_Demo.ipynb ✅
```

---

## Recommended Actions

### Immediate (Before v1.0)

1. **Update README.md** - Clarify which files are current vs legacy
2. **Add deprecation notices** to legacy files or move to `legacy/` folder
3. **Clean up test databases** - Many `AIM26-Test*` databases in root
4. **Remove workspace pickle files** - `workspace_variables_*.pkl`

### Short Term

1. **Create `notebooks/` directory** - Move/create notebooks using SMART library
2. **Create `scripts/` directory** - Move CLI tools and create new utility scripts
3. **Create `data/` directory** - Organize input data files
4. **Create `legacy/` directory** - Move deprecated files (optional - or just delete)

### Medium Term

1. **Migrate remaining legacy notebooks** to use SMART library
2. **Delete legacy parquet/pickle files** after verifying SQLite migration
3. **Clean up year-specific viewers** - Use configurable `session_viewer_app.py`

---

## Summary Counts

| Category | Current | Legacy | Data | Total |
|----------|---------|--------|------|-------|
| Streamlit Apps | 3 | 2 | - | 5 |
| CLI Tools | 2 | 1 | - | 3 |
| Notebooks | 1 | 5+ | - | 6+ |
| Library Modules | 8 | - | - | 8 |
| Databases | - | - | 20+ | 20+ |
| Data Files | - | - | 30+ | 30+ |

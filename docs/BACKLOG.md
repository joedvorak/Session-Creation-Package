# SMART Project Backlog

## Priority Levels

| Priority | Description |
|----------|-------------|
| **P0** | Critical - Blocks core functionality, must fix before release |
| **P1** | High - Important for usability, should fix before release |
| **P2** | Medium - Improves experience, nice to have for release |
| **P3** | Low - Future enhancement, can defer |

## Status Key

| Status | Description |
|--------|-------------|
| 🔴 Not Started | Work has not begun |
| 🟡 In Progress | Currently being worked on |
| 🟢 Complete | Finished and tested |
| ⏸️ Blocked | Waiting on dependency |
| 🔵 Deferred | Pushed to future phase |

---

## Category: Embedding Cache & Model Consistency

### EMB-001: Fix Cache Status Display
**Priority**: P0 | **Status**: 🔴 Not Started

**Problem**: The embedding step UI does not correctly show cached embedding counts. It always shows 0 cached even when embeddings exist.

**Root Cause**: The cache lookup uses `EmbeddingConfig(model_name=X, model_version=X)` but embeddings are stored with actual `model_version` from the embedder (e.g., Ollama returns digest hash, Gemini may return API version).

**Solution**: 
1. Add `get_coverage_by_model(texts)` method to `EmbeddingCache` that returns cached counts per model configuration
2. Redesign UI to show available cached models with coverage counts
3. User selects a cached model config to load or generate missing

**Files**: 
- `smart/core/database.py` - Add coverage query method
- `smart_app.py` - Redesign embedding step UI

**Acceptance Criteria**:
- [ ] UI shows list of models with cached embeddings for current presentations
- [ ] Each model shows X/Y cached count
- [ ] User can select a model to load from cache or generate missing
- [ ] No API key required just to check cache status

---

### EMB-002: Track Model Configuration Across Workflow
**Priority**: P0 | **Status**: 🔴 Not Started

**Problem**: The system doesn't ensure consistent model configuration is used across all embeddings. New presentations, committee descriptions, or updates could use different models, making similarity comparisons invalid.

**Solution**:
1. Store "active embedding config" in session state when loading from cache
2. When generating new embeddings, require using the same config as cached
3. If cached model is unavailable (e.g., Ollama model deleted), allow load but block generation
4. Display warning if trying to mix incompatible embeddings

**Files**:
- `smart_app.py` - Track active config, validate on generation
- `smart/core/database.py` - Add methods to query stored configs

**Acceptance Criteria**:
- [ ] Active model config stored after loading/generating embeddings
- [ ] New embedding generation uses same config as loaded
- [ ] Clear warning if model not available for generation
- [ ] Cannot proceed with mixed embedding models

---

### EMB-003: Support Loading Without Available Model
**Priority**: P1 | **Status**: 🔴 Not Started

**Problem**: If a model is updated, deleted, or unavailable, users cannot load their cached embeddings.

**Solution**:
1. Allow loading from cache even if model not currently available
2. Display warning that new embeddings cannot be generated

---

### EMB-005: Remove SentenceTransformers Backend
**Priority**: P1 | **Status**: 🔴 Not Started

**Problem**: SentenceTransformers backend doesn't work well in the Streamlit app. Ollama provides a better local model experience with more flexibility.

**Rationale**:
- Ollama handles local models better
- Reduces dependencies
- Simplifies codebase
- SentenceTransformers adds complexity without benefit

**Files to Update**:
- `smart/llm/embeddings.py` - Remove `SentenceTransformerEmbedder` class
- `smart/llm/__init__.py` - Remove exports if any
- `smart_app.py` - Remove ST from backend options
- `docs/ARCHITECTURE.md` - Update embedding backends table
- `requirements.txt` - Remove sentence-transformers dependency

**Acceptance Criteria**:
- [ ] `SentenceTransformerEmbedder` class removed
- [ ] No references to sentence-transformers in codebase
- [ ] `sentence-transformers` removed from requirements.txt
- [ ] Documentation updated
- [ ] Existing Gemini and Ollama backends still work
3. Track which presentations have embeddings vs which don't
4. Block generation steps if model unavailable

**Acceptance Criteria**:
- [ ] Can load cached embeddings without model availability
- [ ] Warning shown if model unavailable for new embeddings
- [ ] Clear indication of which items cannot be embedded

---

### EMB-004: Query Available Task Types
**Priority**: P3 | **Status**: 🔴 Not Started

**Problem**: Different models may support different task types for embeddings. Currently hardcoded list.

**Solution**: 
- For Gemini: Query API for supported task types
- For Ollama: Task type not applicable (remove from UI for Ollama)
- For Sentence-Transformers: Not applicable

**Acceptance Criteria**:
- [ ] Task type dropdown populated from model capabilities
- [ ] Hidden/disabled when not applicable

---

## Category: Data Management & Editing

### DATA-001: Implement Data Editing Mode
**Priority**: P1 | **Status**: 🟡 In Progress

**Problem**: Users cannot edit presentations, sessions, or committees within the app. Must use external tools.

**Current State**: Edit mode toggle added, basic structure in place, but functionality incomplete.

**Remaining Work**:
- [ ] Test edit presentation form
- [ ] Test delete presentation with confirmation
- [ ] Test move presentation between sessions
- [ ] Test edit session title
- [ ] Test delete session
- [ ] Verify metrics recalculation after edits

**Files**: `smart_app.py`, `smart/core/database.py`

---

### DATA-002: Metrics Recalculation Button
**Priority**: P1 | **Status**: 🟢 Complete

**Problem**: After editing session membership, coherence/distinctiveness metrics are stale.

**Solution**: 
- Added `clear_session_metrics()` on edit operations
- Added "Recalculate Metrics" button in data viewer
- Calculates both coherence AND distinctiveness

**Verification Needed**: Test with actual edits

---

### DATA-003: Add New Presentation Manually
**Priority**: P2 | **Status**: 🔴 Not Started

**Problem**: Late additions or corrections require re-importing entire file.

**Solution**:
1. Add form to create single presentation
2. Mark as needing embedding
3. Display warning until embedded

**Acceptance Criteria**:
- [ ] Form to add presentation with required fields
- [ ] New presentation marked for embedding
- [ ] Shown in "needs embedding" list

---

### DATA-004: Add New Committee Manually
**Priority**: P2 | **Status**: 🔴 Not Started

Similar to DATA-003 but for committees.

---

## Category: Export & Import

### EXP-001: Viewer Bundle Filename Consistency
**Priority**: P1 | **Status**: 🟢 Complete

**Problem**: Exporter used `presentation_similarities.parquet` but viewer expected `pres_similarities.parquet`.

**Solution**: Fixed exporter to use consistent naming. Added fallback in viewer for backwards compatibility.

---

### EXP-002: Session Export Column Names
**Priority**: P1 | **Status**: 🟢 Complete

**Problem**: Export used `coherence` and `presentation_count` but viewer expected `session_coherence` and `session_size`.

**Solution**: Added column renames in `export_sessions()`. Viewer handles both old and new names.

---

### EXP-003: Incremental Export for Updates
**Priority**: P2 | **Status**: 🔴 Not Started

**Problem**: After changes, must re-export entire bundle. Would be nice to update in place.

**Solution**: Consider delta export or in-place updates for viewer bundle.

---

## Category: User Interface

### UI-001: Clickable Wizard Steps
**Priority**: P1 | **Status**: 🟢 Complete

**Problem**: Steps in sidebar were static text, required "Jump to Step" dropdown.

**Solution**: Steps are now clickable buttons with completion status (✅/👉/○).

---

### UI-002: Conference Name Display
**Priority**: P1 | **Status**: 🟢 Complete

**Problem**: Unclear which conference data is loaded.

**Solution**: Conference name shown above "View Current Data" button.

---

### UI-003: Persistent Error Messages
**Priority**: P1 | **Status**: 🟢 Complete

**Problem**: Error messages (e.g., "embeddings not loaded") disappeared on rerun.

**Solution**: Store errors in session state, display with dismiss button.

---

### UI-004: Ollama Model Auto-Discovery
**Priority**: P1 | **Status**: 🟢 Complete

**Problem**: Users had to manually type Ollama model names.

**Solution**: 
- Query `/api/tags` for available models
- Show in selectbox instead of text input
- Filter to embedding models for embedding step
- Filter out embedding-only models for title generation step

---

### UI-005: Ollama Connection Check
**Priority**: P1 | **Status**: 🟢 Complete

**Problem**: No feedback if Ollama server not running.

**Solution**: Check connection before showing model list, display error if unreachable.

---

### UI-006: Text Truncation Handling for Embeddings
**Priority**: P1 | **Status**: 🟢 Complete

**Problem**: Ollama embedding fails with context length exceeded error.

**Solution**:
- Auto-truncate text on error with retry
- Learn max length for future texts
- Display warning showing which texts were truncated

---

## Category: Testing Infrastructure

### TEST-001: Set Up Pytest Infrastructure
**Priority**: P0 | **Status**: 🔴 Not Started

**Problem**: No automated tests. All testing is manual through the UI.

**Solution**:
1. Create `tests/` directory structure
2. Add pytest and pytest-cov to requirements
3. Create conftest.py with fixtures
4. Add CI configuration (GitHub Actions)

**Files to Create**:
- `tests/conftest.py` - Shared fixtures
- `tests/test_database.py` - Database tests
- `tests/test_embeddings.py` - Embedding backend tests
- `tests/test_placement.py` - Clustering algorithm tests
- `tests/test_metrics.py` - Metrics calculation tests
- `pyproject.toml` or `pytest.ini` - Test configuration

**Acceptance Criteria**:
- [ ] `pytest` runs successfully from project root
- [ ] Test database fixtures (in-memory SQLite)
- [ ] Mock embeddings for deterministic tests
- [ ] Coverage report generated

---

### TEST-002: Database Unit Tests
**Priority**: P0 | **Status**: 🔴 Not Started

**Tests Needed**:
- [ ] EmbeddingCache: store/retrieve single embedding
- [ ] EmbeddingCache: batch operations
- [ ] EmbeddingCache: config hash consistency
- [ ] EmbeddingCache: coverage query (new method)
- [ ] ConferenceDB: CRUD presentations
- [ ] ConferenceDB: CRUD sessions
- [ ] ConferenceDB: CRUD committees
- [ ] ConferenceDB: placements and moves
- [ ] ConferenceDB: metrics updates

---

### TEST-003: Embedding Backend Tests
**Priority**: P1 | **Status**: 🔴 Not Started

**Tests Needed**:
- [ ] Mock embedder for tests (deterministic vectors)
- [ ] CachedEmbedder cache hit/miss behavior
- [ ] OllamaEmbedder truncation logic
- [ ] Model version tracking
- [ ] Batch embedding consistency

---

### TEST-004: Placement Algorithm Tests
**Priority**: P1 | **Status**: 🔴 Not Started

**Tests Needed**:
- [ ] Minimum session size respected
- [ ] Maximum sessions limit
- [ ] Hybrid session handling
- [ ] Coherence calculation correctness
- [ ] Deterministic results for same input

---

### TEST-005: Integration Tests
**Priority**: P2 | **Status**: 🔴 Not Started

**Tests Needed**:
- [ ] Full workflow: import → embed → cluster → export
- [ ] Re-import with cache reuse
- [ ] Export bundle can be loaded by viewer

---

## Category: Documentation & Project Organization

### DOC-001: Create ARCHITECTURE.md
**Priority**: P0 | **Status**: 🟢 Complete

**Purpose**: Living technical reference documenting current system state. Must be updated concurrently with code changes.

**Content**:
- [x] Component overview (SMART library modules)
- [x] Application descriptions (which app does what)
- [x] Data flow diagrams
- [x] Database schemas (cache and working DBs)
- [x] Export format specifications
- [x] File status table (current vs legacy vs deprecated)
- [x] Deployment pattern for standalone viewers

**Key Requirement**: This document tracks what exists NOW, not what's planned. Update when code changes.

**Files**: `docs/ARCHITECTURE.md`

**Completed**: 2026-02-12

---

### DOC-002: Create CONTRIBUTING.md
**Priority**: P0 | **Status**: 🟢 Complete

**Purpose**: Define process rules for making changes to ensure documentation stays in sync with code.

**Content**:
- [x] Pre-change checklist (what to review before starting)
- [x] Documentation-code sync requirements
- [x] When to update which documents
- [x] Test requirements before committing
- [x] Commit message conventions
- [x] How to add new features vs fix bugs

**Key Rules Documented**:
1. Code changes → Update ARCHITECTURE.md
2. New features → Update ROADMAP.md phase status
3. Bug fixes → Update BACKLOG.md status
4. Before merging → Run tests

**Files**: `docs/CONTRIBUTING.md`

**Completed**: 2026-02-12

---

### DOC-003: Execute Reorganization Plan
**Priority**: P0 | **Status**: 🟢 Complete

**Purpose**: Restructure project files for clarity. See [Reorganization Plan](#reorganization-plan) section below.

**Phases**:
1. ✅ Create directory structure
2. ✅ Move current files to appropriate locations
3. ✅ Mark/move legacy files
4. ✅ Clean up data files
5. ✅ Update documentation with new paths

**Dependencies**: DOC-001 (ARCHITECTURE.md) should be created first to document current state before reorganizing.

**Acceptance Criteria**:
- [x] Clear separation between current and legacy code
- [x] Data files organized
- [x] Test databases cleaned up
- [x] All current apps still work after moves (verified imports)

**Completed**: February 2026

**New Structure**:
```
apps/               - Streamlit apps (smart_app.py, session_viewer_app.py, session_progress_tracker.py)
scripts/            - CLI tools (smart_cli.py, migrate_data.py)
notebooks/          - Current notebooks (SMART_Demo.ipynb)
data/               - Input data (submissions/, committees/, hybrid/)
databases/          - SQLite files
legacy/             - Deprecated code and notebooks
tests/              - Test suite (planned)
```

---

### DOC-004: README Update
**Priority**: P2 | **Status**: 🔴 Not Started

**Note**: Deferred until current version stabilizes. Current README documents legacy system which is still accurate for that code.

**Sections Needed** (when ready):
- [ ] Project overview and purpose
- [ ] Installation instructions
- [ ] Quick start guide
- [ ] Configuration (API keys, Ollama setup)
- [ ] CLI usage
- [ ] Streamlit app usage
- [ ] Clear note about legacy vs current files

---

### DOC-005: API Documentation
**Priority**: P2 | **Status**: 🔴 Not Started

**Modules to Document**:
- [ ] smart/core/database.py
- [ ] smart/core/placement.py
- [ ] smart/llm/embeddings.py
- [ ] smart/llm/titles.py

---

## Category: Scripts & Notebooks

### SCRIPT-001: Core Workflow Scripts
**Priority**: P1 | **Status**: 🔴 Not Started

**Purpose**: Provide repeatable, automated workflows for testing and demonstration.

**Scripts to Create**:

| Script | Description | Priority |
|--------|-------------|----------|
| `scripts/embed_presentations.py` | Generate and cache embeddings | P1 |
| `scripts/create_sessions.py` | Run clustering with parameters | P1 |
| `scripts/generate_titles.py` | Generate LLM titles for sessions | P1 |
| `scripts/export_bundle.py` | Export viewer bundle | P1 |
| `scripts/full_workflow.py` | End-to-end pipeline | P1 |

**Common CLI Pattern**:
```bash
python scripts/embed_presentations.py \
    --conference AIM2026 \
    --input submissions.csv \
    --backend ollama \
    --model nomic-embed-text-v2-moe
```

**Acceptance Criteria**:
- [ ] Each script has `--help` with argument documentation
- [ ] Scripts use SMART library (not reimplementing logic)
- [ ] Configurable via command-line arguments
- [ ] Clear success/error output
- [ ] Can be chained together for full workflow

---

### SCRIPT-002: Utility Scripts
**Priority**: P2 | **Status**: 🔴 Not Started

**Scripts to Create**:

| Script | Description |
|--------|-------------|
| `scripts/compare_models.py` | Compare embedding models on same dataset |
| `scripts/validate_cache.py` | Check cache integrity and coverage |
| `scripts/migration.py` | Migrate between database versions |
| `scripts/stats.py` | Print database statistics |

---

### NB-001: Core Workflow Notebooks
**Priority**: P1 | **Status**: 🔴 Not Started

**Purpose**: Documented, step-by-step illustrations for developers and documentation.

**Notebooks to Create**:

| Notebook | Description |
|----------|-------------|
| `notebooks/01_data_import.ipynb` | Loading and exploring presentation data |
| `notebooks/02_embedding_generation.ipynb` | Embedding backends, caching, similarity |
| `notebooks/03_duplicate_detection.ipynb` | Finding near-duplicates via cosine similarity |
| `notebooks/04_session_clustering.ipynb` | Agglomerative clustering with constraints |
| `notebooks/05_metrics_visualization.ipynb` | Coherence, distinctiveness, outliers |
| `notebooks/06_title_generation.ipynb` | LLM title and keyword generation |

**Each Notebook Should Include**:
- [ ] Purpose and context explanation
- [ ] Required imports and setup
- [ ] Step-by-step code with explanatory markdown
- [ ] Visualizations where applicable
- [ ] Expected output examples
- [ ] Links to related notebooks

**Acceptance Criteria**:
- [ ] Notebooks run cleanly from top to bottom
- [ ] Sample data included or generated
- [ ] Can run without API keys (using mock/cached data)
- [ ] Markdown explains the "why" not just the "what"

---

### NB-002: Advanced Notebooks
**Priority**: P2 | **Status**: 🔴 Not Started

**Notebooks to Create**:

| Notebook | Description |
|----------|-------------|
| `notebooks/07_committee_matching.ipynb` | Committee-to-session similarity matching |
| `notebooks/08_full_pipeline.ipynb` | Complete workflow demonstration |
| `notebooks/09_model_comparison.ipynb` | Comparing embedding models |
| `notebooks/10_parameter_tuning.ipynb` | Exploring clustering parameters |

---

### NB-003: Research Notebooks
**Priority**: P3 | **Status**: 🔴 Not Started

**Purpose**: Support research paper reproducibility.

**Notebooks to Create**:
- [ ] Reproduce AIM 2025 ITSC study results
- [ ] Generate figures for publications
- [ ] Statistical analysis of session quality

---

## Category: Future Features

### FUT-001: Session Similarity for Scheduling
**Priority**: P3 | **Status**: 🔵 Deferred

**Description**: Use session-to-session similarities to help avoid scheduling similar sessions at the same time.

---

### FUT-002: Attendee Topic Search
**Priority**: P3 | **Status**: 🔵 Deferred

**Description**: Allow attendees to search for presentations by entering a topic description. Would require embedding API access with rate limiting.

**Constraints**: Cost control for API usage.

---

### FUT-003: Presentation Move Preview
**Priority**: P3 | **Status**: 🔵 Deferred

**Description**: Before moving a presentation, preview what the new coherence/distinctiveness would be for affected sessions.

---

### FUT-004: LLM Title Refinement
**Priority**: P3 | **Status**: 🔵 Deferred

**Description**: Use secondary LLM pass to reduce title repetitiveness across all sessions (e.g., too many "Advances in..." titles).

---

## Reorganization Plan

### Target Directory Structure

```
Session-Creation-Package/
├── smart/                      # Core library (no change)
│   ├── __init__.py
│   ├── core/
│   │   ├── database.py
│   │   ├── placement.py
│   │   └── metrics.py
│   ├── io/
│   │   ├── loaders.py
│   │   └── exporters.py
│   └── llm/
│       ├── embeddings.py
│       └── titles.py
│
├── apps/                       # NEW: Streamlit applications
│   ├── smart_app.py           # Main session creation wizard
│   ├── session_viewer_app.py  # Bundle viewer
│   └── session_progress_tracker.py  # Progress tracking
│
├── scripts/                    # NEW: CLI tools
│   ├── smart_cli.py           # Main CLI
│   ├── migrate_data.py        # Data migration
│   └── (future workflow scripts)
│
├── notebooks/                  # NEW: Current notebooks using SMART library
│   └── SMART_Demo.ipynb       # (move from root)
│
├── data/                       # NEW: Input data files
│   ├── submissions/           # Raw submission files
│   ├── committees/            # Committee data
│   └── hybrid/                # Hybrid session assignments
│
├── output/                     # Export outputs (exists)
│   └── (bundles, exports)
│
├── databases/                  # NEW: Working databases
│   └── (*.db files - organized by conference)
│
├── tests/                      # NEW: Test suite
│   ├── conftest.py
│   ├── test_database.py
│   └── ...
│
├── docs/                       # Documentation (exists)
│   ├── ROADMAP.md
│   ├── BACKLOG.md
│   ├── TESTING.md
│   ├── FILE_INVENTORY.md
│   ├── ARCHITECTURE.md        # NEW
│   └── CONTRIBUTING.md        # NEW
│
├── legacy/                     # NEW: Deprecated files (optional)
│   ├── session_organizer.py
│   ├── session_creation_app_v2.py
│   ├── session_creation_viewer_web_app.py
│   ├── session_creation_viewer_web_app26.py
│   └── notebooks/
│       ├── session creation operation.ipynb
│       └── ...
│
├── .env                        # Environment config
├── .gitignore
├── README.md
└── requirements.txt
```

### Migration Steps

#### Phase 1: Create Structure (No Breaking Changes)
1. Create empty directories: `apps/`, `scripts/`, `notebooks/`, `data/`, `databases/`, `tests/`, `legacy/`
2. Create placeholder README in each new directory explaining purpose

#### Phase 2: Move Current Files
| Source | Destination | Notes |
|--------|-------------|-------|
| `smart_app.py` | `apps/smart_app.py` | Update launch scripts |
| `session_viewer_app.py` | `apps/session_viewer_app.py` | |
| `session_progress_tracker.py` | `apps/session_progress_tracker.py` | |
| `smart_cli.py` | `scripts/smart_cli.py` | |
| `migrate_data.py` | `scripts/migrate_data.py` | |
| `SMART_Demo.ipynb` | `notebooks/SMART_Demo.ipynb` | |

#### Phase 3: Move Legacy Files
| Source | Destination | Notes |
|--------|-------------|-------|
| `session_organizer.py` | `legacy/session_organizer.py` | |
| `session_creation_app_v2.py` | `legacy/session_creation_app_v2.py` | |
| `session_creation_viewer_web_app.py` | `legacy/session_creation_viewer_web_app.py` | |
| `session_creation_viewer_web_app26.py` | `legacy/session_creation_viewer_web_app26.py` | |
| `session creation operation.ipynb` | `legacy/notebooks/` | |
| `session creation operation 2026_old.ipynb` | `legacy/notebooks/` | |
| `session creation operation 2026_simple.ipynb` | `legacy/notebooks/` | |
| `SMART 2026 Initial Sort.ipynb` | `legacy/notebooks/` | Evaluate first |
| `Processing Steps.ipynb` | `legacy/notebooks/` | Evaluate first |
| `Data Export.ipynb` | `legacy/notebooks/` | Evaluate first |

#### Phase 4: Organize Data Files
| Pattern | Destination | Notes |
|---------|-------------|-------|
| `*.db` | `databases/` | Group by conference |
| `Submissions_*.csv/xlsx` | `data/submissions/` | |
| `ASABE Committees*.csv` | `data/committees/` | |
| `*Hybrid*.xlsx` | `data/hybrid/` | |
| `abstracts*.xlsx` | `data/submissions/` | |

#### Phase 5: Clean Up
| Files | Action |
|-------|--------|
| `workspace_variables_*.pkl` | Delete |
| `*_backup.parquet` | Delete after verifying SQLite has data |
| `embeddings_backup.pkl` | Delete after verifying |
| Legacy parquet files used only by legacy viewers | Keep in `legacy/` or delete |

### Import Path Updates

After moving apps to `apps/` directory, launch commands change:

**Before**:
```bash
streamlit run smart_app.py
streamlit run session_viewer_app.py
```

**After**:
```bash
streamlit run apps/smart_app.py
streamlit run apps/session_viewer_app.py
```

Alternatively, create launcher scripts in root:
```bash
# run_smart.sh
#!/bin/bash
streamlit run apps/smart_app.py "$@"
```

### Verification Checklist

After each phase, verify:
- [ ] All current apps still launch and function
- [ ] Import paths resolve correctly
- [ ] No broken references in documentation
- [ ] Git status shows expected changes only

---

## Completed Items Archive

| ID | Description | Completed |
|----|-------------|-----------|
| EXP-001 | Viewer bundle filename consistency | 2026-02 |
| EXP-002 | Session export column names | 2026-02 |
| UI-001 | Clickable wizard steps | 2026-02 |
| UI-002 | Conference name display | 2026-02 |
| UI-003 | Persistent error messages | 2026-02 |
| UI-004 | Ollama model auto-discovery | 2026-02 |
| UI-005 | Ollama connection check | 2026-02 |
| UI-006 | Text truncation handling | 2026-02 |
| DATA-002 | Metrics recalculation button | 2026-02 |

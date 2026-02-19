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
**Priority**: P0 | **Status**: 🟢 Complete

**Problem**: The embedding step UI does not correctly show cached embedding counts. It always shows 0 cached even when embeddings exist.

**Root Cause**: The cache lookup uses `EmbeddingConfig(model_name=X, model_version=X)` but embeddings are stored with actual `model_version` from the embedder (e.g., Ollama returns digest hash, Gemini may return API version).

**Solution**: 
1. Add `get_coverage_by_model(texts)` method to `EmbeddingCache` that returns cached counts per model configuration
2. Redesign UI to show available cached models with coverage counts
3. User selects a cached model config to load or generate missing

**Files**: 
- `smart/core/database.py` - Added `get_coverage_by_model()` and `get_all_model_configs()` methods
- `apps/smart_app.py` - Redesigned embedding step UI with model selection

**Acceptance Criteria**:
- [x] UI shows list of models with cached embeddings for current presentations
- [x] Each model shows X/Y cached count
- [x] User can select a model to load from cache or generate missing
- [x] No API key required just to check cache status

**Implementation**:
- `EmbeddingCache.get_coverage_by_model(texts)` queries all model configs and returns coverage for each
- UI shows radio buttons for available cached models plus "Generate New" option
- Selected model's config stored in `st.session_state.active_embedding_config`
- Model details shown in expandable section

**Completed**: 2026-02-12

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

---

### EMB-006: Fix Config Hash Mutation Bug
**Priority**: P1 | **Status**: 🔴 Not Started

**Problem**: `store_embedding()` and `store_embeddings_batch()` mutate the passed `EmbeddingConfig` object by setting `config.dimensions`. This causes:
1. Config hash to change after storage
2. Mismatch between hash stored in embeddings table vs hash computed from model_registry
3. `get_coverage_by_model()` returns empty results because it uses registry dimensions

**Root Cause**: Line in `store_embedding()`: `config.dimensions = dimensions`

**Solution Options**:
1. **Don't include dimensions in config_hash()** - simplest, may affect model version tracking
2. **Don't mutate config** - compute hash before setting dimensions
3. **Store original hash** - store config_hash computed before dimensions mutation

**Acceptance Criteria**:
- [ ] Config hash consistent between store and retrieve
- [ ] `get_coverage_by_model()` returns correct results
- [ ] Test `test_get_coverage_by_model` passes (currently skipped)

**Files**: `smart/core/database.py`

**Discovered**: 2026-02-13 during TEST-001 implementation

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

## Category: Placement Algorithm

### PLACE-001: Fix Final Filling Process Overloading Last Sessions
**Priority**: P0 | **Status**: � Complete

**Problem**: During the final filling process of bottom-up hierarchical clustering, most remaining presentations were being placed into the last few sessions created rather than distributed across related sessions throughout the hierarchy. This resulted in unbalanced session sizes with final sessions having many more presentations than intended.

**Root Cause**: Two issues in `_bottom_up_cluster` and `_assign_remaining_items`:
1. A `max_session_size` override forced session creation even past `merge_stop_index`, creating extra sessions beyond the intended stop point.
2. `_assign_remaining_items` enforced `max_session_size` and created new single-item overflow sessions when all existing sessions were full. These single-item sessions became magnets for subsequent remaining items, concentrating items in tail sessions.

**Solution** (matching legacy `create_sessions_w_hybrid` behavior):
1. Removed the `max_session_size` override that created sessions past `merge_stop_index`. Sessions now only finalize when `i < merge_stop_index AND len(final_clusters) < max_sessions`.
2. Changed `_assign_remaining_items` to always assign to the most similar existing cluster regardless of `max_session_size`, never creating new sessions. Items go where they semantically belong.

**Before/After Benchmark** (AIM26, 1150 presentations, min=8, max=12):

| Metric | Before | After | Legacy |
|--------|--------|-------|--------|
| Tail/Body ratio | 1.46 (IMBALANCED) | 1.00 (BALANCED) | 0.87 |
| Size CV | 0.221 | 0.215 | 0.264 |
| Max size | 17 | 20 | 28 |
| Coherence mean | 0.8644 | 0.8620 | 0.8608 |
| Fit min | 0.7557 | 0.7827 | 0.7813 |
| Sessions > 150% mean | 4 | 1 | 3 |

**Files**: `smart/core/placement.py` — both `OralSessionPlacement` and `HybridFirstPlacement`

**Completed**: 2026-02-18

---

### PLACE-002: Max Session Size Not Enforced
**Priority**: P2 | **Status**: 🔵 Deferred

**Problem**: The max session size parameter in the UI is not enforced as a hard limit. During the final filling phase, remaining presentations are assigned to their best-fit session regardless of size (matching legacy behavior).

**Context**: PLACE-001 fix intentionally removed `max_session_size` enforcement from `_assign_remaining_items` to prevent tail-overloading. The legacy algorithm also had no max enforcement. Enforcing a hard max would require a redistribution pass after the final fill, which could hurt coherence.

**Current State**:
- `max_session_size` is used during tree traversal to trigger early finalization
- Final fill ignores max size (items go to most similar session)
- Typical overshoot is modest (max 20 with target 12 on AIM26 data)

**Possible Future Approaches**:
1. Post-fill redistribution: Move items from oversized sessions to next-best session
2. Iterative balancing: Trade items between neighboring sessions to reduce max
3. Soft limit with penalty: Weight similarity against session size during fill
4. Accept as-is: Document that max_session_size is advisory, not enforced

**Acceptance Criteria**:
- [ ] Either: Max size enforced with acceptable coherence tradeoff
- [ ] Or: UI control relabeled as "target" with tooltip explaining behavior

**Files**: 
- `smart/core/placement.py` - Would need post-fill balancing pass
- `apps/smart_app.py` - Relabel UI control

---

### PLACE-003: Per-Session Dendrograms for Visualization and Ordering
**Priority**: P2 | **Status**: 🔴 Not Started

**Problem**: There is no way to visualize the internal structure of a session — which presentations are most similar to each other, how they cluster within the session, or what a good presentation order might be. This information is valuable both for verifying session quality and for helping organizers set presentation order.

**Approach**: Recompute a linkage matrix from the session's presentation embeddings after placement (rather than extracting subtrees from the original placement linkage tree). This is the better approach because:
1. **Simpler**: Just call `scipy.cluster.hierarchy.linkage()` on the subset of embeddings for each session. No need to store/track the original tree or handle the complexity of sessions composed from multiple tree branches.
2. **More useful**: The resulting dendrogram directly shows within-session relationships, which is what organizers need for ordering. It closely approximates how early sessions were formed from the original tree, while remaining meaningful for later sessions that drew from multiple branches.
3. **Data already available**: Embeddings are accessible in all three apps — `st.session_state.embeddings` in smart_app.py and session_progress_tracker.py, and the existing (but unused) `load_embeddings()` in session_viewer_app.py.

**Implementation Plan**:

1. **Utility function** in `smart/core/metrics.py` or new `smart/core/visualization.py`:
   - `compute_session_dendrogram(embeddings, indices, labels)` → returns matplotlib Figure
   - Uses `scipy.cluster.hierarchy.linkage()` + `dendrogram()` on session-subset embeddings
   - Labels leaf nodes with presentation titles (truncated)
   - Returns figure for embedding in Streamlit via `st.pyplot()`

2. **Integration points** (in priority order):
   - **`apps/session_viewer_app.py`** (`show_sessions_tab`): Add dendrogram when a session is selected. This is the primary organizer tool and the highest-value integration. Wire up the existing `load_embeddings()` function (currently unused).
   - **`apps/smart_app.py`** (`_render_sessions_tab`): Add dendrogram in the session detail view. Embeddings already in session state.
   - **`apps/session_progress_tracker.py`**: Add dendrogram in session detail. Embeddings already in session state.

3. **Rendering**: `matplotlib` with `scipy.cluster.hierarchy.dendrogram()` rendered via `st.pyplot()`. Matplotlib is already used in the benchmark scripts. Consider `plotly` for interactive dendrograms as a future enhancement.

**Acceptance Criteria**:
- [ ] Selecting a session in the viewer shows a dendrogram of its presentations
- [ ] Leaf labels show truncated presentation titles
- [ ] Dendrogram renders correctly for sessions of size 8–20
- [ ] Available in session_viewer_app.py and smart_app.py session detail views

**Files**:
- `smart/core/metrics.py` or `smart/core/visualization.py` - Dendrogram computation utility
- `apps/session_viewer_app.py` - Wire up `load_embeddings()`, add dendrogram to session detail
- `apps/smart_app.py` - Add dendrogram to `_render_sessions_tab()` session detail

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
**Priority**: P0 | **Status**: 🟢 Complete

**Problem**: No automated tests. All testing is manual through the UI.

**Solution**:
1. Create `tests/` directory structure
2. Add pytest and pytest-cov to requirements
3. Create conftest.py with fixtures
4. Add CI configuration (GitHub Actions) - deferred

**Files Created**:
- `tests/conftest.py` - Shared fixtures (mock embedder, sample data, temp DBs)
- `tests/test_database.py` - Database tests (21 tests)
- `pytest.ini` - Test configuration

**Acceptance Criteria**:
- [x] `pytest` runs successfully from project root
- [x] Test database fixtures (in-memory SQLite)
- [x] Mock embeddings for deterministic tests
- [ ] Coverage report generated (pytest-cov installed, not configured)
- [ ] CI configuration (GitHub Actions) - deferred

**Completed**: 2026-02-13

---

### TEST-002: Database Unit Tests
**Priority**: P0 | **Status**: 🟡 In Progress

**Tests Needed**:
- [x] EmbeddingCache: store/retrieve single embedding
- [x] EmbeddingCache: batch operations
- [x] EmbeddingCache: config hash consistency (partial - see EMB-006)
- [ ] EmbeddingCache: coverage query (blocked by EMB-006)
- [x] ConferenceDB: CRUD presentations
- [x] ConferenceDB: CRUD sessions
- [ ] ConferenceDB: CRUD committees
- [ ] ConferenceDB: placements and moves
- [x] ConferenceDB: metrics updates

**Note**: One test skipped due to config hash bug (EMB-006)

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

### TEST-006: Placement Algorithm Regression Tests
**Priority**: P0 | **Status**: � Complete

**Problem**: Need to verify placement algorithm produces correct, consistent results. Current manual testing revealed issues with final filling process (see PLACE-001).

**Approach**: Created synthetic test embeddings with known cluster structure for deterministic testing.

**Test Cases Implemented** (in `tests/test_placement.py`):
1. ✅ **Determinism**: `test_oral_placement_deterministic`, `test_oral_placement_deterministic_with_different_seed`
2. ✅ **Session Balance**: `test_session_size_distribution_reasonable`, `test_final_filling_distributes_evenly`
3. ⚠️ **Max Size Enforcement**: `test_no_session_exceeds_max_size` - XFAIL, catches PLACE-002 bug
4. ✅ **No Tiny Sessions**: `test_no_tiny_sessions_after_placement`
5. ✅ **Coherence Quality**: `test_clustered_items_placed_together`

**Additional Tests**:
- Edge cases: single item, fewer than min, exact fit
- Hybrid placement: fills first, preserves assignments
- Metadata validation: counts, result structure
- Performance: scales to 500 items

**Test Results**:
- 14 passed, 1 xfailed (PLACE-002 bug confirmed)
- Runs in 0.14s with no external dependencies
- Deterministic using seeded random embeddings

**Acceptance Criteria**:
- [x] Test data bundle created from real conference → Used synthetic clustered embeddings instead
- [x] Baseline session assignments stored → Tests verify determinism directly
- [x] Test detects regressions in placement behavior
- [x] Test fails if final filling concentrates in last sessions → test passes currently; may fail after algorithm changes
- [x] Can run without API access (uses cached embeddings)

**Completed**: Session

**Files Created**:
- `tests/test_placement.py` - 15 regression tests

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
**Priority**: P2 | **Status**: � Complete

**Completed**: All project READMEs updated to reflect current v2.0.0 codebase.

**Sections Updated**:
- [x] Project overview and purpose
- [x] Installation instructions
- [x] Quick start guide
- [x] Configuration (API keys, Ollama setup)
- [x] CLI usage
- [x] Streamlit app usage
- [x] Clear note about legacy vs current files

**Files Updated**:
- `README.md` — Complete rewrite covering current architecture, placement strategies, quality metrics, embedding backends, database schema, benchmarking, and testing
- `scripts/README.md` — Added benchmark scripts (run_benchmark.py, placement_benchmark.py, extract_benchmark_fixture.py)
- `tests/README.md` — Complete rewrite reflecting implemented test suite (51 tests, fixtures, markers)

---

### DOC-005: API Documentation
**Priority**: P2 | **Status**: � Complete

**Completed**: All inline docstrings expanded to include Args/Returns. Standalone API reference generated.

**Modules** (7 total, all well-documented inline):

| Module | Lines | Public API | Coverage |
|--------|------:|:----------:|:--------:|
| `smart/core/database.py` | 2214 | ~50 items | 100% |
| `smart/core/placement.py` | 1087 | ~21 items | 100% |
| `smart/core/metrics.py` | 402 | 7 items | 100% |
| `smart/io/loaders.py` | 563 | ~15 items | 100% |
| `smart/io/exporters.py` | 1130 | ~18 items | 100% |
| `smart/llm/embeddings.py` | 693 | ~30 items | 100% |
| `smart/llm/titles.py` | 485 | ~20 items | 100% |

**Completed Tasks**:
- [x] Add docstrings to `EmbeddingConfig.to_dict` and `EmbeddingConfig.from_dict`
- [x] Expand one-line docstrings on ~8 `ConferenceDB` getter methods with Args/Returns
- [x] Generate standalone API reference → `docs/API_REFERENCE.md` (1555 lines)

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
|----|-------------|----------|
| DOC-005 | API documentation (inline + reference) | 2025-02-19 |
| DOC-004 | README update for v2.0.0 | 2025-02-19 |
| PLACE-001 | Fix final filling process overloading last sessions | 2025-02-18 |
| EMB-001 | Fix cache status display | 2025-02-12 |
| EXP-001 | Viewer bundle filename consistency | 2026-02 |
| EXP-002 | Session export column names | 2026-02 |
| UI-001 | Clickable wizard steps | 2026-02 |
| UI-002 | Conference name display | 2026-02 |
| UI-003 | Persistent error messages | 2026-02 |
| UI-004 | Ollama model auto-discovery | 2026-02 |
| UI-005 | Ollama connection check | 2026-02 |
| UI-006 | Text truncation handling | 2026-02 |
| DATA-002 | Metrics recalculation button | 2026-02 |
| TEST-001 | Set up pytest infrastructure | 2026-02-13 |

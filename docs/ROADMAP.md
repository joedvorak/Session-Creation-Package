# SMART - Session Matching And Automated Recommendation Tool

## Project Roadmap

### Vision

SMART automates the initial session creation process for ASABE's Annual International Meeting (AIM), replacing ineffective presenter self-selection with AI-driven thematic clustering. The system leverages text embedding models to measure semantic similarity between presentation abstracts and uses agglomerative clustering to form coherent sessions that meet size constraints. LLMs generate session title suggestions and keywords to accelerate the human organization process.

**Key Insight**: The goal is not to automatically create final sessions, but to provide session organizers with a strong starting point—a core cluster of 8-10 highly related submissions plus similar alternatives—so they can focus on curation and audience engagement rather than scrambling to fill undersubscribed sessions or cut from oversubscribed ones.

---

## Project Phases

### Phase 1: Core Session Creation (✅ Complete)
**Status**: Functional

The core algorithm and workflow for initial session organization:
- [x] Import presentation data from CSV/Excel
- [x] Generate text embeddings (title + abstract)
- [x] Identify and remove duplicate/near-duplicate submissions
- [x] Agglomerative clustering with session size constraints
- [x] Support for hybrid/invited sessions (pre-assigned presentations)
- [x] Calculate session coherence and distinctiveness metrics
- [x] LLM-based title and keyword generation
- [x] Committee-to-session matching based on similarity
- [x] Export to spreadsheets for manual review

### Phase 2: Viewer Application for Organizers (✅ Complete)
**Status**: Functional

Web-based viewer for session organizers to review and refine sessions:
- [x] Session list with coherence/distinctiveness metrics
- [x] Presentation details within each session
- [x] Similarity-based presentation search
- [x] Sort/filter by various metrics
- [x] Export bundle format for sharing

### Phase 3: Stability & Testing (🔄 In Progress)
**Status**: Current Focus

Ensure reliability and maintainability:
- [ ] Automated testing infrastructure (pytest)
- [ ] Embedding cache consistency fixes
- [ ] Model configuration tracking across workflow
- [ ] Error handling improvements
- [ ] Documentation completion
- [ ] Python scripts for repeatable workflows
- [ ] Jupyter notebooks for developer documentation

### Phase 4: Iterative Updates for Organizers (📋 Planned)
**Status**: Design Phase

Support for the ongoing organization process as the conference evolves:
- [ ] Incremental embedding generation for new/updated presentations
- [ ] Metrics recalculation after manual presentation moves
- [ ] Track presentation withdrawals and session changes
- [ ] Re-export updated bundles for viewer
- [ ] Maintain cache compatibility across updates

### Phase 5: Attendee-Facing Viewer (📋 Planned)
**Status**: Future

Final conference schedule viewer for attendees:
- [ ] Finalized session and presentation data
- [ ] Find similar presentations/sessions from any presentation
- [ ] Text search for presentation topics
- [ ] Session schedule integration
- [ ] (Future/Constrained) Topic search via embedding API with rate limiting

---

## Use Cases & Workflows

### 1. Initial Session Creation (Organizer)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INITIAL SESSION CREATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Import  │ → │ Generate │ → │  Remove  │ → │  Create  │ → │ Generate │  │
│  │   Data   │   │Embeddings│   │Duplicates│   │ Sessions │   │  Titles  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│       │                                             │              │        │
│       ▼                                             ▼              ▼        │
│  ┌──────────┐                                 ┌──────────┐   ┌──────────┐  │
│  │  Import  │                                 │  Match   │   │  Export  │  │
│  │ Hybrids  │                                 │Committees│   │  Bundle  │  │
│  └──────────┘                                 └──────────┘   └──────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Inputs**:
- Presentation CSV/Excel (title, abstract, submission ID, presenter info)
- Optional: Hybrid session assignments (invited presentations)
- Optional: Committee list with descriptions

**Outputs**:
- Working database with sessions and placements
- Spreadsheet export for manual review and portal data entry
- Viewer bundle for interactive exploration

### 2. Ongoing Organization (Session Organizers)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ONGOING ORGANIZATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     Viewer Application                              │   │
│   │  • Review assigned sessions                                         │   │
│   │  • See presentation similarities                                    │   │
│   │  • Search for replacements when speakers withdraw                   │   │
│   │  • View coherence/distinctiveness metrics                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                  Manual Updates in Portal                           │   │
│   │  • Move presentations between sessions                              │   │
│   │  • Accept/reject presentations                                      │   │
│   │  • Finalize session titles                                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     Re-import & Refresh                             │   │
│   │  • Import updated presentation list                                 │   │
│   │  • Recalculate metrics for changed sessions                         │   │
│   │  • Generate new embeddings only for new/changed presentations       │   │
│   │  • Re-export viewer bundle                                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Attendee Discovery (Future)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ATTENDEE DISCOVERY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   Final Schedule Viewer                             │   │
│   │  • Browse sessions by time/room/topic                               │   │
│   │  • View presentation details                                        │   │
│   │  • "Find similar" from any presentation                             │   │
│   │  • Text search for keywords                                         │   │
│   │  • (Future) Topic search via embedding query                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Note: Topic search via API would require rate limiting to control costs  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Components

### Core Library (`smart/`)

| Module | Purpose |
|--------|---------|
| `smart/core/database.py` | SQLite databases for embeddings cache and working data |
| `smart/core/placement.py` | Clustering strategies (agglomerative, hybrid-first) |
| `smart/core/metrics.py` | Coherence, distinctiveness, outlier detection |
| `smart/llm/embeddings.py` | Embedding backends (Gemini, Ollama, Sentence-Transformers) |
| `smart/llm/titles.py` | Title/keyword generation (Gemini, Ollama) |
| `smart/io/loaders.py` | CSV/Excel import with column mapping |
| `smart/io/exporters.py` | Spreadsheet and viewer bundle export |

### Applications

| Application | Purpose |
|-------------|---------|
| `smart_app.py` | Main Streamlit wizard for session creation |
| `session_viewer_app.py` | Streamlit viewer for exploring sessions |
| `session_organizer.py` | CLI tool for batch processing |
---

## Deliverables

The SMART project provides functionality in three complementary forms, each serving different use cases:

### 1. Streamlit Applications

**Purpose**: Interactive web UI for users who need visual feedback and guided workflows.

| Application | Use Case |
|-------------|----------|
| `smart_app.py` | Full workflow wizard for session creation |
| `session_viewer_app.py` | Interactive session exploration and review |

**Best For**:
- Session organizers exploring options
- Real-time parameter tuning
- Viewing metrics and similarity relationships
- Non-programmers

### 2. Python Scripts

**Purpose**: Repeatable, automated workflows for testing and demonstration.

| Script | Description |
|--------|-------------|
| `scripts/embed_presentations.py` | Generate and cache embeddings for presentations |
| `scripts/create_sessions.py` | Run clustering with specified parameters |
| `scripts/generate_titles.py` | Generate LLM titles for all sessions |
| `scripts/export_bundle.py` | Export viewer bundle from working database |
| `scripts/full_workflow.py` | End-to-end pipeline from import to export |
| `scripts/compare_models.py` | Compare embedding models on same dataset |
| `scripts/validate_cache.py` | Check cache integrity and coverage |

**Best For**:
- Quick reproducible testing
- Benchmarking different configurations
- Demonstrations without UI overhead
- CI/CD pipelines
- Batch processing

### 3. Jupyter Notebooks

**Purpose**: Documented, step-by-step illustrations for developers and documentation.

| Notebook | Description |
|----------|-------------|
| `notebooks/01_data_import.ipynb` | Loading and exploring presentation data |
| `notebooks/02_embedding_generation.ipynb` | Embedding backends, caching, similarity |
| `notebooks/03_duplicate_detection.ipynb` | Finding near-duplicates via cosine similarity |
| `notebooks/04_session_clustering.ipynb` | Agglomerative clustering with constraints |
| `notebooks/05_metrics_visualization.ipynb` | Coherence, distinctiveness, outliers |
| `notebooks/06_title_generation.ipynb` | LLM title and keyword generation |
| `notebooks/07_committee_matching.ipynb` | Committee-to-session similarity matching |
| `notebooks/08_full_pipeline.ipynb` | Complete workflow demonstration |

**Best For**:
- Developer onboarding
- Algorithm explanation with visualizations
- Research documentation
- Interactive exploration with inline results
- Sharing with collaborators

### Relationship Between Deliverables

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SMART Library (smart/)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   database   │  │  embeddings  │  │   placement  │  │   exporters  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │  Streamlit    │       │    Python     │       │    Jupyter    │
    │  Applications │       │    Scripts    │       │   Notebooks   │
    └───────────────┘       └───────────────┘       └───────────────┘
    Interactive UI          Automation/Testing       Documentation
```
### Data Storage

| Database | Purpose |
|----------|---------|
| `{conference}_cache.db` | Embedding cache with model/version tracking |
| `{conference}_working.db` | Presentations, sessions, placements, titles |

---

## Non-Goals (Explicitly Out of Scope)

1. **Replacing human judgment**: The system provides starting points, not final decisions
2. **Automatic peer review**: Unlike some systems, ASABE doesn't require peer review for acceptance
3. **Real-time collaborative editing**: Organizers use external tools (spreadsheets, portal) for coordination
4. **Unlimited API access for attendees**: Cost constraints prevent open embedding queries

---

## Success Criteria for v1.0 Release

- [ ] All Phase 1 and 2 features stable and documented
- [ ] Automated tests for core functionality
- [ ] Embedding cache works correctly across sessions
- [ ] Model configuration tracked to ensure consistency
- [ ] Clear error messages for common issues
- [ ] README with installation and usage instructions
- [ ] Successfully used for AIM 2026 organization

---

## References

- Dvorak, J. (2025). "Optimizing ASABE AIM Session Creation with Text Clustering and LLMs"
- Dvorak, J. (2025). "AI Tools and Text Embedding for Session Organization at ASABE's Annual International Meeting"
- Project Visualization: https://sessioncreation-aim2025-itsc.streamlit.app/

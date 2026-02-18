# SMART — Session Matching And Automated Recommendation Tool

A tool suite for automatically creating and organizing technical sessions for academic conferences using AI-powered similarity analysis. SMART uses embedding models to group presentations by topic similarity, supports hybrid sessions with pre-assigned invited speakers, and provides interactive tools for session review and visualization.

Developed for the ASABE (American Society of Agricultural and Biological Engineers) Annual International Meeting, but designed to generalize to other academic conferences.

## Overview

SMART automates the traditionally manual process of organizing conference presentations into thematic sessions:

1. **Import** presentation titles and abstracts from CSV/Excel
2. **Embed** text using Gemini, Ollama, or SentenceTransformers
3. **Cluster** presentations into coherent sessions via hierarchical clustering
4. **Fill hybrid sessions** with pre-assigned invited speakers, topped up with similar content
5. **Generate titles** and keywords for sessions using LLMs
6. **Match sessions to committees** for review assignment
7. **Export** for organizer review or cloud-based viewer

## Repository Structure

```
smart/                          # Core library (v2.0)
├── core/
│   ├── database.py             # SQLite persistence (EmbeddingCache, ConferenceDB)
│   ├── placement.py            # Session assignment algorithms
│   └── metrics.py              # Session quality metrics
├── io/
│   ├── loaders.py              # File loading with column mapping
│   └── exporters.py            # Export with redaction profiles
└── llm/
    ├── embeddings.py           # Gemini, Ollama, SentenceTransformers backends
    └── titles.py               # LLM title/keyword generation

apps/                           # Streamlit applications
├── smart_app.py                # Main session creation wizard
├── session_viewer_app.py       # Read-only session/presentation viewer
└── session_progress_tracker.py # Progress tracking from existing assignments

scripts/                        # CLI tools and utilities
├── smart_cli.py                # End-to-end pipeline CLI
├── migrate_data.py             # Import legacy data into SQLite
├── run_benchmark.py            # Placement strategy benchmarking
├── placement_benchmark.py      # Benchmark framework (metrics, reports)
└── extract_benchmark_fixture.py # Extract test data from databases

tests/                          # pytest test suite
├── conftest.py                 # Shared fixtures
├── test_database.py            # EmbeddingCache and ConferenceDB tests
├── test_placement.py           # Placement algorithm tests (synthetic data)
├── test_placement_real.py      # Real-data regression tests (AIM26)
└── fixtures/                   # Test data (*.npz, gitignored)

notebooks/                      # Jupyter notebooks
└── SMART_Demo.ipynb            # Interactive demo of the SMART library

databases/                      # SQLite databases (gitignored)
data/                           # Input data (submissions, committees, hybrid)
output/                         # Generated output (sessions, bundles, benchmarks)
legacy/                         # Deprecated v1.x code (reference only)
docs/                           # Architecture, backlog, roadmap
```

## Installation

### Requirements

Python 3.10+ recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `streamlit`, `google-genai`, `sentence-transformers`, `python-dotenv`, `tqdm`.

### Environment Setup

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

The Gemini API key is required for embedding generation (gemini-embedding-001) and title generation. Free tier is sufficient for typical conference volumes (~1000 presentations).

### Optional: Ollama for Local LLMs

For local embedding and title generation without API keys:

1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull nomic-embed-text` (embeddings) or `ollama pull llama3.2` (titles)
3. Ollama serves automatically after installation

## Quick Start

### Streamlit App (Recommended)

The main workflow is a guided wizard:

```bash
streamlit run apps/smart_app.py
```

Steps: Conference Setup → Import Presentations → Generate Embeddings → Create Sessions → Generate Titles → Match Committees → Review & Export

### CLI Pipeline

For scripted or batch processing:

```bash
python scripts/smart_cli.py \
    -p data/submissions/Submissions.csv \
    -H data/hybrid/hybrid_sessions.csv \
    -c data/committees/ASABE\ Committees.csv \
    -o output/ \
    -n AIM2026 \
    -y 26 \
    --min-size 9 \
    --max-size 12
```

### Session Viewer

To explore sessions after creation:

```bash
streamlit run apps/session_viewer_app.py
```

Load a viewer bundle directory (exported from the main app) containing parquet files and optional embeddings.

## Applications

| App | Purpose | Launch |
|-----|---------|--------|
| `smart_app.py` | Full session creation wizard | `streamlit run apps/smart_app.py` |
| `session_viewer_app.py` | Read-only session explorer | `streamlit run apps/session_viewer_app.py` |
| `session_progress_tracker.py` | Track progress from existing assignments | `streamlit run apps/session_progress_tracker.py` |

## Core Concepts

### Placement Strategies

SMART provides multiple session creation algorithms:

| Strategy | Description |
|----------|-------------|
| `OralSessionPlacement` | Bottom-up hierarchical clustering. Finalizes sessions when clusters reach minimum size, allowing popular topics to spawn multiple sessions. |
| `HybridFirstPlacement` | Same as oral, but fills pre-assigned hybrid sessions with similar content first. |
| `TraditionalClusterPlacement` | Standard fcluster cut at a fixed tree level. Provided for comparison. |
| `PosterThematicOrdering` | Orders posters by similarity without session boundaries. |

### Quality Metrics

| Metric | Description |
|--------|-------------|
| **Session Coherence** | Mean pairwise cosine similarity within a session (higher = more focused) |
| **Session Distinctiveness** | Silhouette score measuring separation from other sessions |
| **Presentation Fit** | How well each presentation matches its assigned session |

### Embedding Backends

| Backend | Model Example | Dimensions | Notes |
|---------|--------------|------------|-------|
| Gemini | `gemini-embedding-001` | 3072 | Highest quality, requires API key |
| Ollama | `nomic-embed-text` | 768 | Local, no API key |

### Database Schema

SMART uses two SQLite databases per conference:

- **`*_cache.db`**: Embedding cache keyed by text hash + model config. Prevents re-computation.
- **`*_working.db`**: Conference data — presentations, sessions, placements, committees, metrics.

## Benchmarking

Compare placement strategies on real conference data:

```bash
python scripts/run_benchmark.py \
    --strategies hybrid_first oral traditional legacy \
    --merge-stops 0.90 0.95 0.98
```

Generates an HTML report with comparison tables and plots (session size distribution, coherence by creation order, fit distribution). Reports are self-contained with embedded images.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_placement.py -v

# Run with coverage
python -m pytest tests/ --cov=smart --cov-report=term-missing
```

Test suite: 51 tests covering database operations, placement algorithms (synthetic and real data), determinism, and session quality metrics.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture and module design
- [docs/BACKLOG.md](docs/BACKLOG.md) — Issue tracker and feature backlog
- [docs/ROADMAP.md](docs/ROADMAP.md) — Development phases and milestones
- [docs/TESTING.md](docs/TESTING.md) — Test strategy and conventions

## Legacy Code

The `legacy/` directory contains the original v1.x implementation:

- `session_organizer.py` — Original algorithmic engine (replaced by `smart/` package)
- `session_creation_app_v2.py` — Tkinter desktop app (replaced by `apps/smart_app.py`)
- `session_creation_viewer_web_app.py` — Original Streamlit viewer (replaced by `apps/session_viewer_app.py`)

Legacy code is retained for reference and benchmark comparison. See [legacy/README.md](legacy/README.md).

## License

This project is developed for academic and conference organization purposes.

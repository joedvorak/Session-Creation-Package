# Example Data

Deidentified example data from the ASABE AIM 2026 conference, provided so that
fresh installations can verify functionality and run the full test suite without
requiring access to the original (confidential) submission data.

## Deidentification

The following personally identifiable information (PII) was removed or replaced
before committing this data:

- **Author / presenter names** — replaced with generic placeholders
- **Email addresses** — removed
- **Submission ownership details** — anonymized

Abstract text, titles, keywords, and all algorithmic outputs (embeddings,
sessions, metrics) are preserved unchanged.

## Contents

| Path | Size | Description |
|------|------|-------------|
| `databases/AIM26_Example_cache.db` | 18 MB | Embedding cache — 1,234 cached embeddings (3072-dim, `gemini-embedding-001`) |
| `databases/AIM26_Example_working.db` | 6.9 MB | Conference database — 1,160 presentations, 97 sessions (6 hybrid) |
| `submissions/AIM26_Example_Submission_Data.xlsx` | 1.8 MB | Raw submission spreadsheet (deidentified) |
| `hybrid/AIM26_Example_Hybrid_Sessions.xlsx` | 28 KB | Hybrid session pre-assignments |
| `exports/20260220/` | 9 MB | Full organizer export (parquet + metadata) |
| `exports/20260220_presenter_list.xlsx` | 1.1 MB | Presenter list export |
| `exports/20260220_room_assignment.xlsx` | 80 KB | Room assignment export |

**Total: ~37 MB**

## Usage

### Running the test suite

The regression tests in `tests/test_placement_real.py` automatically detect
these example databases and use them when the pre-extracted `.npz` fixture is
not present. On a fresh clone:

```bash
python -m pytest tests/ -v
```

All tests should run without skips (except those marked `xfail` for known bugs).

### Generating the benchmark fixture (optional)

For faster repeated test runs, you can pre-extract embeddings into an `.npz`
file:

```bash
python scripts/extract_benchmark_fixture.py
```

This reads from the example databases and writes
`tests/fixtures/aim26_benchmark.npz`. Subsequent test runs will load from the
`.npz` instead of querying SQLite.

### Exploring the data

Load the example databases in the Streamlit app or in a notebook:

```python
from smart.core.database import ConferenceDB, EmbeddingCache

db = ConferenceDB("examples/databases/AIM26_Example_working.db")
cache = EmbeddingCache("examples/databases/AIM26_Example_cache.db")

presentations = db.get_presentations()
print(f"Loaded {len(presentations)} presentations")
```

Or explore the export files directly:

```python
import pandas as pd

sessions = pd.read_parquet("examples/exports/20260220/sessions.parquet")
presentations = pd.read_parquet("examples/exports/20260220/presentations.parquet")
```

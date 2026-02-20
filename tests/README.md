# Tests

Test suite for the SMART library.

## Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `test_database.py` | 22 | EmbeddingCache CRUD, ConferenceDB CRUD, config hashing, batch ops |
| `test_placement.py` | 17 | Determinism, balance, edge cases, hybrid, coherence, performance |
| `test_placement_real.py` | 12 | Real AIM26 data: determinism, PLACE-001/002 regression, size distribution, quality |
| `conftest.py` | — | Shared fixtures (MockEmbedder, sample presentations, embeddings) |

**Total: 51 tests** (47 pass, 2 skip for EMB-006, 2 xfail for PLACE-002)

## Fixtures & Example Data

Real-data regression tests (`test_placement_real.py`) load benchmark data from
one of two sources, checked in order:

1. **`fixtures/aim26_benchmark.npz`** — Pre-extracted embeddings (12.4 MB,
   gitignored). Fastest loading. Generate with:
   ```bash
   python scripts/extract_benchmark_fixture.py
   ```
2. **`../examples/databases/AIM26_Example_*.db`** — Deidentified example
   databases shipped with the repository. Used automatically when the `.npz`
   is absent. No setup required on a fresh clone.

On a fresh clone, all 51 tests run without skips (except 2 `xfail` for known
bugs).

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_placement.py -v

# Run with coverage
python -m pytest tests/ --cov=smart --cov-report=term-missing
```

## Markers

Custom markers defined in `pytest.ini`:

- `@pytest.mark.unit` — Fast, no external dependencies
- `@pytest.mark.integration` — Requires database or file I/O
- `@pytest.mark.slow` — Long-running tests
- `@pytest.mark.requires_ollama` — Requires local Ollama server
- `@pytest.mark.requires_gemini` — Requires Gemini API key

See `docs/TESTING.md` for the full testing strategy.

# Tests

Test suite for the SMART library.

## Status

Test infrastructure is planned but not yet implemented. See BACKLOG.md items TEST-001 through TEST-005.

## Planned Structure

```
tests/
├── conftest.py          # Shared fixtures
├── test_database.py     # EmbeddingCache and ConferenceDB tests
├── test_embeddings.py   # Embedding backend tests
├── test_placement.py    # Clustering algorithm tests
├── test_metrics.py      # Metrics calculation tests
├── test_loaders.py      # Import functionality tests
├── test_exporters.py    # Export functionality tests
├── fixtures/            # Sample data files
└── mocks/               # Mock implementations
```

## Running Tests

Once implemented:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=smart --cov-report=html
```

See `docs/TESTING.md` for the full testing strategy.

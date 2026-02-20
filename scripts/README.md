# SMART Scripts

Command-line tools for batch processing, benchmarking, and utilities.

## Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `smart_cli.py` | Full workflow CLI | `python scripts/smart_cli.py --help` |
| `migrate_data.py` | Migrate legacy data to SQLite | `python scripts/migrate_data.py --help` |
| `run_benchmark.py` | Compare placement strategies | `python scripts/run_benchmark.py --help` |
| `placement_benchmark.py` | Benchmark framework (metrics, reports) | Imported by `run_benchmark.py` |
| `extract_benchmark_fixture.py` | Extract test fixtures from databases | `python scripts/extract_benchmark_fixture.py` |

## Usage

From the project root:

```bash
# Run full workflow
python scripts/smart_cli.py \
    --presentations data/submissions/abstracts.xlsx \
    --output output/ \
    --conference AIM2026

# Migrate existing parquet data
python scripts/migrate_data.py \
    --source presentations_with_embeddings.parquet \
    --target AIM2026

# Run placement benchmark (generates HTML report)
python scripts/run_benchmark.py \
    --strategies hybrid_first oral traditional legacy \
    --merge-stops 0.90 0.95 0.98
```

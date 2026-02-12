# SMART Scripts

Command-line tools for batch processing and utilities.

## Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `smart_cli.py` | Full workflow CLI | `python scripts/smart_cli.py --help` |
| `migrate_data.py` | Migrate legacy data to SQLite | `python scripts/migrate_data.py --help` |

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
```

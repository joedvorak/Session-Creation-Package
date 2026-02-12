# Databases

SQLite databases used by SMART applications.

## Database Types

| Pattern | Purpose |
|---------|---------|
| `*_cache.db` | Embedding cache (reusable across workflows) |
| `*_working.db` | Conference working data (presentations, sessions, etc.) |

## Naming Convention

Databases are named by conference: `{conference_name}_{type}.db`

Examples:
- `AIM2026_cache.db` - Embedding cache for AIM 2026
- `AIM2026_working.db` - Working data for AIM 2026

## Cleanup

Test databases can be safely deleted. Production databases should be backed up 
before deletion.

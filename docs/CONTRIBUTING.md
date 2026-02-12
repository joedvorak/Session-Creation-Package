# Contributing to SMART

**Last Updated**: February 2026

This document defines the process for making changes to the SMART project. Following these guidelines ensures documentation stays in sync with code and changes don't break existing functionality.

---

## Golden Rules

1. **Document before or with code changes** - Never commit code without updating relevant documentation
2. **Test before committing** - Run affected tests and verify apps still work
3. **One concern per change** - Don't mix unrelated changes
4. **Update the backlog** - Mark tasks as in-progress when starting, complete when done

---

## Change Types & Checklists

### Bug Fix

When fixing a bug:

- [ ] Identify the bug in BACKLOG.md (or add it if not present)
- [ ] Mark the backlog item as 🟡 In Progress
- [ ] Make the code fix
- [ ] Test the fix manually
- [ ] Add automated test if applicable
- [ ] Mark the backlog item as 🟢 Complete
- [ ] Update ARCHITECTURE.md only if the fix changes system behavior

### New Feature

When adding a new feature:

- [ ] Ensure feature is in BACKLOG.md with acceptance criteria
- [ ] Mark as 🟡 In Progress
- [ ] Implement the feature
- [ ] Update ARCHITECTURE.md with new components/behaviors
- [ ] Update ROADMAP.md phase checklist if applicable
- [ ] Add tests for new functionality
- [ ] Test manually in affected apps
- [ ] Mark as 🟢 Complete in BACKLOG.md

### Refactoring

When restructuring code without changing behavior:

- [ ] Document the refactoring goal
- [ ] Update ARCHITECTURE.md if module structure changes
- [ ] Run all tests before and after
- [ ] Verify all apps still work
- [ ] Update FILE_INVENTORY.md if files move

### Removing Code

When removing features or dependencies:

- [ ] Create backlog item if not present
- [ ] List all files that need updating
- [ ] Remove the code
- [ ] Update ARCHITECTURE.md to remove references
- [ ] Update requirements.txt if removing dependencies
- [ ] Search codebase for remaining references
- [ ] Run tests to verify nothing broken
- [ ] Update FILE_INVENTORY.md

---

## Documentation Sync Requirements

### When to Update Each Document

| Document | Update When... |
|----------|----------------|
| **ARCHITECTURE.md** | Adding/removing modules, changing data flow, modifying schemas, adding/removing apps |
| **BACKLOG.md** | Starting work, completing work, finding bugs, adding features |
| **ROADMAP.md** | Completing phase milestones, changing project direction |
| **FILE_INVENTORY.md** | Adding/removing/moving files |
| **TESTING.md** | Adding test categories, changing test infrastructure |
| **README.md** | Major releases, installation changes (defer until stable) |

### Documentation Update Examples

**Adding a new embedding backend:**
```
1. BACKLOG.md - Mark task in progress
2. smart/llm/embeddings.py - Add the class
3. ARCHITECTURE.md - Add to embedding backends table
4. BACKLOG.md - Mark complete
```

**Fixing a UI bug:**
```
1. BACKLOG.md - Add bug if not present, mark in progress
2. smart_app.py - Fix the bug
3. BACKLOG.md - Mark complete
(No ARCHITECTURE.md update needed - behavior unchanged)
```

**Removing SentenceTransformers:**
```
1. BACKLOG.md - Mark EMB-005 in progress
2. smart/llm/embeddings.py - Remove class
3. requirements.txt - Remove dependency
4. ARCHITECTURE.md - Remove from backends table
5. FILE_INVENTORY.md - Update if files removed
6. BACKLOG.md - Mark complete
```

---

## Code Standards

### Python Style

- Follow PEP 8
- Use type hints for function signatures
- Docstrings for public functions and classes
- Keep functions focused and small

### Imports

```python
# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import streamlit as st

# SMART library
from smart.core.database import EmbeddingCache
from smart.llm.embeddings import create_embedder
```

### Error Handling

- Use specific exceptions, not bare `except:`
- Provide helpful error messages
- In Streamlit apps, use `st.error()` for user-facing errors
- Log errors for debugging

### Session State (Streamlit)

- Initialize all session state variables in one place
- Use consistent naming: `st.session_state.embeddings`, not `st.session_state["emb"]`
- Document what each session state variable holds

---

## Testing Requirements

### Before Committing

1. **Run affected tests** (when test infrastructure exists):
   ```bash
   pytest tests/test_affected_module.py
   ```

2. **Manual verification** for UI changes:
   - Launch the affected app
   - Test the changed functionality
   - Test one happy path through the workflow

### Test Coverage Goals

| Category | Requirement |
|----------|-------------|
| Database operations | Must have unit tests |
| Metrics calculations | Must have unit tests |
| Embedding backends | Integration tests with mocks |
| UI components | Manual testing (automated later) |

---

## Commit Messages

### Format

```
<type>: <short description>

<optional longer description>

<optional: Fixes #issue or Closes BACKLOG-ID>
```

### Types

| Type | Use For |
|------|---------|
| `feat` | New features |
| `fix` | Bug fixes |
| `refactor` | Code restructuring |
| `docs` | Documentation only |
| `test` | Adding/updating tests |
| `chore` | Maintenance (dependencies, config) |

### Examples

```
feat: Add Ollama model auto-discovery

Query /api/tags endpoint to list available models.
Show selectbox instead of text input when Ollama connected.

Closes EMB-004
```

```
fix: Correct cache status display

The cache lookup was using model_name for both name and version,
causing mismatches with actual stored embeddings.

Fixes EMB-001
```

```
docs: Update ARCHITECTURE.md with viewer deployment pattern
```

---

## File Organization

### Current Structure (see FILE_INVENTORY.md for details)

```
Session-Creation-Package/
├── smart/              # Core library - DO NOT put apps here
├── docs/               # All documentation
├── output/             # Export outputs (gitignored data)
├── exports/            # Spreadsheet exports (gitignored data)
├── *.py                # Applications and scripts (root level for now)
└── *.ipynb             # Notebooks (root level for now)
```

### Planned Structure (see BACKLOG DOC-003)

After reorganization, apps move to `apps/`, scripts to `scripts/`, etc.

### Where to Put New Files

| File Type | Location |
|-----------|----------|
| New SMART library module | `smart/<category>/` |
| New Streamlit app | Root (or `apps/` after reorg) |
| New CLI script | Root (or `scripts/` after reorg) |
| New test | `tests/` |
| New documentation | `docs/` |

---

## Deployment Artifacts

### Standalone Viewers

Year-specific viewer files in the root are **deployment artifacts**, not active development files:

- `session_creation_viewer_web_app.py` - AIM 2025 base
- `session_creation_viewer_web_app26.py` - AIM 2026 base

**Do not modify these** unless updating for a new deployment. The canonical viewer is `session_viewer_app.py`.

### Creating a New Deployment

1. Export bundle from smart_app or session_progress_tracker
2. Create new repository for deployment
3. Copy `session_viewer_app.py` to new repo
4. Copy bundle contents to `data/` in new repo
5. Add `requirements.txt` and `.streamlit/secrets.toml`
6. Deploy to Streamlit Cloud

---

## Getting Help

- **Architecture questions**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **What tasks exist**: See [BACKLOG.md](BACKLOG.md)
- **Project direction**: See [ROADMAP.md](ROADMAP.md)
- **File purposes**: See [FILE_INVENTORY.md](FILE_INVENTORY.md)
- **Test strategy**: See [TESTING.md](TESTING.md)

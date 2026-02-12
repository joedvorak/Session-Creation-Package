# Legacy Files

**⚠️ DEPRECATED**: These files are from the original SMART implementation before the library refactor.

## Status

These files depend on `session_organizer.py`, the original monolithic implementation. 
They have been replaced by the SMART library (`smart/`) and new applications in `apps/`.

## Contents

### Python Files

| File | Original Purpose | Replaced By |
|------|------------------|-------------|
| `session_organizer.py` | Core algorithm engine | `smart/` library |
| `session_creation_app_v2.py` | Tkinter desktop GUI | `apps/smart_app.py` |
| `session_creation_viewer_web_app.py` | AIM 2025 viewer | `apps/session_viewer_app.py` |
| `session_creation_viewer_web_app26.py` | AIM 2026 viewer | `apps/session_viewer_app.py` |

### Notebooks

Located in `notebooks/` subdirectory. These use `session_organizer.py` and have been 
replaced by notebooks using the SMART library.

## Do Not Use

These files are kept for reference only. Do not use them for new work.
Use the SMART library and apps in the main directories instead.

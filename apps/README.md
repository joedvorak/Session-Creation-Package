# SMART Applications

Streamlit applications for session organization.

## Applications

| File | Purpose | Launch Command |
|------|---------|----------------|
| `smart_app.py` | Main session creation wizard | `streamlit run apps/smart_app.py` |
| `session_viewer_app.py` | Session and presentation viewer | `streamlit run apps/session_viewer_app.py` |
| `session_progress_tracker.py` | Progress tracking for existing sessions | `streamlit run apps/session_progress_tracker.py` |

## Quick Start

From the project root:

```bash
# Session creation (Phase 1-3)
streamlit run apps/smart_app.py

# View exported sessions (Phase 2+)
streamlit run apps/session_viewer_app.py

# Track progress for existing sessions (Phase 4)
streamlit run apps/session_progress_tracker.py
```

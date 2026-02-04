"""
SMART Session Viewer App

A Streamlit web application for exploring session organization results.
Loads data from a ViewerBundle (directory or zip file).

Usage:
    streamlit run session_viewer_app.py
    
Or with a specific bundle:
    streamlit run session_viewer_app.py -- --bundle ./AIM2026_bundle

For deployment, place a bundle at one of these default locations:
    ./data/          (directory with manifest.json)
    ./viewer_data/   (directory with manifest.json)  
    ./data.zip       (zipped bundle)
"""

import numpy as np
import streamlit as st
import pandas as pd
import hmac
from pathlib import Path
from typing import Optional

# ============================================================
# CONFIGURATION - Set default bundle path for deployment
# ============================================================
# When deploying, set this to your bundle path (relative to app directory)
# Leave as None to auto-detect or require user input
DEFAULT_BUNDLE_PATH = "data"  # Default: look for ./data/ directory
# ============================================================

# Page config must be first Streamlit command
st.set_page_config(
    page_title="SMART Session Viewer",
    page_icon=":material/category_search:",
    layout="wide",
)


# ============================================================
# Data Loading Functions
# ============================================================

@st.cache_data
def load_bundle_manifest(bundle_path: str) -> dict:
    """Load and parse the bundle manifest."""
    import json
    bundle_path = Path(bundle_path)
    manifest_path = bundle_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_presentations(bundle_path: str, use_encrypted: bool = False, password: Optional[str] = None) -> pd.DataFrame:
    """Load presentations from bundle."""
    bundle_path = Path(bundle_path)
    
    if use_encrypted and password:
        encrypted_path = bundle_path / "presentations_encrypted.crypt"
        if encrypted_path.exists():
            try:
                import cryptpandas as crp
                return crp.read_encrypted(path=str(encrypted_path), password=password)
            except Exception:
                pass
    
    # Fall back to public presentations
    pres_path = bundle_path / "presentations.parquet"
    return pd.read_parquet(pres_path)


@st.cache_data
def load_sessions(bundle_path: str) -> pd.DataFrame:
    """Load sessions from bundle."""
    return pd.read_parquet(Path(bundle_path) / "sessions.parquet")


@st.cache_data
def load_pres_similarities(bundle_path: str) -> pd.DataFrame:
    """Load presentation similarity matrix."""
    bundle = Path(bundle_path)
    # Try standard name first, then fallback to legacy name
    pres_sim_path = bundle / "pres_similarities.parquet"
    if not pres_sim_path.exists():
        pres_sim_path = bundle / "presentation_similarities.parquet"
    return pd.read_parquet(pres_sim_path)


@st.cache_data
def load_session_similarities(bundle_path: str) -> pd.DataFrame:
    """Load session similarity matrix."""
    return pd.read_parquet(Path(bundle_path) / "session_similarities.parquet")


@st.cache_data
def load_embeddings(bundle_path: str) -> tuple:
    """Load embeddings if available."""
    emb_path = Path(bundle_path) / "embeddings.npz"
    if emb_path.exists():
        data = np.load(emb_path, allow_pickle=True)
        return data["embeddings"], list(data["abstract_ids"])
    return None, None


def find_bundle_path() -> Optional[str]:
    """
    Find the bundle path using this priority:
    1. Command line argument (--bundle path)
    2. DEFAULT_BUNDLE_PATH configuration (if it exists)
    3. Auto-detect in current directory
    """
    import sys
    
    # 1. Check command line args (highest priority)
    for i, arg in enumerate(sys.argv):
        if arg == "--bundle" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    
    # 2. Check DEFAULT_BUNDLE_PATH configuration
    if DEFAULT_BUNDLE_PATH:
        default_path = Path(DEFAULT_BUNDLE_PATH)
        # Check if it's a valid bundle (has manifest.json or is a zip)
        if default_path.exists():
            if default_path.is_dir() and (default_path / "manifest.json").exists():
                return str(default_path)
            elif default_path.suffix.lower() == '.zip':
                return str(default_path)
            # Also check if presentations.parquet exists (minimal bundle)
            elif default_path.is_dir() and (default_path / "presentations.parquet").exists():
                return str(default_path)
    
    # 3. Auto-detect: Look for common bundle locations
    cwd = Path.cwd()
    
    # Check common default names
    default_names = ["data", "viewer_data", "bundle", "session_data"]
    for name in default_names:
        path = cwd / name
        if path.is_dir() and ((path / "manifest.json").exists() or (path / "presentations.parquet").exists()):
            return str(path)
    
    # Check for any directory with manifest.json
    for path in cwd.iterdir():
        if path.is_dir() and (path / "manifest.json").exists():
            return str(path)
    
    # Check for .zip bundles
    for pattern in ["data.zip", "*_bundle.zip", "*_data.zip"]:
        for path in cwd.glob(pattern):
            return str(path)
    
    return None


# ============================================================
# UI Components
# ============================================================

def show_metric_descriptions():
    """Display expandable metric descriptions."""
    with st.expander("Similarity Metric Descriptions"):
        st.markdown(
            '*All Similarity Metrics range from 0.0 (no relation) to 1.0 (identical). '
            'Scales are relative and not absolute. They vary by model and cannot be '
            'compared across models.*'
        )
        st.markdown(
            "**Presentation-Session Similarity** or **Presentation Session Fit:** "
            "This *presentation metric* is the average cosine similarity between a "
            "presentation and all others in its assigned session."
        )
        st.latex(r"PSS(p_i) = \frac{1}{|s_j| - 1} \sum_{\substack{p_k \in s_j \\ p_k \neq p_i}} sim(p_i, p_k)")
        
        st.markdown(
            "**Session Coherence:** This *session metric* is the average cosine "
            "similarity between all presentations assigned to the same session."
        )
        st.latex(r"SS(s_j) = \frac{1}{|s_j|(|s_j| - 1)} \sum_{\substack{p_i \in s_j \\ p_k \in s_j \\ p_i \neq p_k}} sim(p_i, p_k)")
        
        st.markdown(
            "**Session Std Dev:** The standard deviation of presentation fit scores "
            "within the session. Useful for identifying sessions with outlier presentations."
        )
        
        st.markdown(
            "**Presentation Raw Deviation:** Difference between a presentation's fit "
            "and its session's coherence."
        )
        st.latex(r"RD(p_i) = PSS(p_i) - SS(s_j)")
        
        st.markdown(
            "**Presentation Standardized Deviation:** Raw deviation divided by session "
            "standard deviation (analogous to z-score)."
        )
        st.latex(r"SD(p_i) = \frac{RD(p_i)}{SSD(s_j)}")
        
        st.markdown(
            "**Session-Session Similarity:** Average similarity between all presentations "
            "in one session with all those in another session."
        )


def show_presentations_tab(df_presentations, df_similarity, session_col="Session Code"):
    """Render the presentations tab."""
    st.header("Presentations")
    
    with st.expander("**Instructions** - Click to expand"):
        st.write(
            "Select a presentation by clicking on the checkbox. You can sort or search the list."
        )
        st.write(
            "Once selected, the ten most similar presentations will appear below."
        )
        st.write(
            "Similarity scores range from 0.0 (not similar) to 1.0 (identical)."
        )
    
    st.write(f"Total presentations: {len(df_presentations)}")
    
    # Configure column display
    column_config = {
        "Abstract ID": st.column_config.NumberColumn(format="%i"),
        "Presentation Session Fit": st.column_config.NumberColumn(format="%.3f"),
        "Session Std Dev": None,  # Hide
        "Presentation Raw Deviation": st.column_config.NumberColumn(format="%.3f"),
        "Presentation Standardized Deviation": st.column_config.NumberColumn(format="%.3f"),
    }
    
    event = st.dataframe(
        df_presentations,
        use_container_width=True,
        hide_index=False,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
    )
    
    if event.selection.rows:
        st.header("Selected Presentation")
        selected_row = df_presentations.iloc[event.selection.rows[0]]
        st.write(f"**{selected_row['Title']}**")
        
        if "Abstract" in df_presentations.columns:
            st.write(selected_row.get("Abstract", ""))
        
        st.header("Most Similar Presentations")
        
        # Get similarity scores for selected presentation
        pres_id = selected_row.name  # Index
        
        if pres_id in df_similarity.index:
            similar = df_similarity.loc[pres_id].sort_values(ascending=False)
            similar = similar.drop(pres_id, errors='ignore')  # Remove self
            
            # Build similar presentations DataFrame
            similar_df = df_presentations.loc[df_presentations.index.isin(similar.index[:20])].copy()
            similar_df = similar_df.loc[similar.index[:20]]  # Reorder by similarity
            similar_df.insert(0, "Similarity Score", similar[:20])
            similar_df.insert(0, "Rank", range(1, len(similar_df) + 1))
            
            st.dataframe(
                similar_df,
                use_container_width=True,
                hide_index=False,
                column_config={
                    "Abstract ID": st.column_config.NumberColumn(format="%i"),
                    "Similarity Score": st.column_config.NumberColumn(format="%.3f"),
                },
            )


def show_sessions_tab(df_sessions, df_presentations, df_session_similarity, session_col="Session Code"):
    """Render the sessions tab."""
    st.header("Sessions")
    
    # Determine session ID column
    sess_id_col = "cluster_id" if "cluster_id" in df_sessions.columns else "session_id"
    
    # Determine size and coherence column names (handle both old and new naming)
    size_col = "session_size" if "session_size" in df_sessions.columns else "presentation_count"
    coherence_col = "session_coherence" if "session_coherence" in df_sessions.columns else "coherence"
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Session Size Distribution")
        if size_col in df_sessions.columns:
            size_data = df_sessions.set_index(sess_id_col)[size_col]
            st.bar_chart(size_data, x_label="Session", y_label="Presentations")
        else:
            st.info("Session size data not available")
    
    with col2:
        st.subheader("Session Coherence Distribution")
        if coherence_col in df_sessions.columns:
            coherence_data = df_sessions.set_index(sess_id_col)[coherence_col]
            st.bar_chart(coherence_data, x_label="Session", y_label="Coherence")
        else:
            st.info("Session coherence data not available")
    
    with st.expander("**Instructions** - Click to expand"):
        st.write(
            "Select a session by clicking on the checkbox. Its presentations and "
            "similar sessions will appear below."
        )
    
    # Session table
    column_config = {
        "session_coherence": st.column_config.NumberColumn(format="%.3f"),
        "coherence": st.column_config.NumberColumn(format="%.3f"),
        "session_std_dev": st.column_config.NumberColumn(format="%.3f"),
    }
    
    event = st.dataframe(
        df_sessions,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
    )
    
    if event.selection.rows:
        selected_session = df_sessions.iloc[event.selection.rows[0]]
        session_id = selected_session[sess_id_col]
        
        st.header(f"Session: {session_id}")
        st.write(f"**Coherence:** {selected_session['session_coherence']:.3f}")
        
        if "title" in selected_session:
            st.write(f"**Title:** {selected_session['title']}")
        
        # Show presentations in this session
        st.subheader("Presentations in Session")
        
        session_pres = df_presentations[df_presentations[session_col] == session_id]
        
        display_cols = ["Presentation Session Fit", "Presentation Standardized Deviation", 
                       "Abstract ID", "Title"]
        if "Abstract" in session_pres.columns:
            display_cols.append("Abstract")
        
        available_cols = [c for c in display_cols if c in session_pres.columns]
        
        st.dataframe(
            session_pres[available_cols] if available_cols else session_pres,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Abstract ID": st.column_config.NumberColumn(format="%i"),
                "Presentation Session Fit": st.column_config.NumberColumn(format="%.3f"),
                "Presentation Standardized Deviation": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        
        # Similar sessions
        st.subheader("Most Similar Sessions")
        
        if session_id in df_session_similarity.columns:
            similar_sessions = df_session_similarity[session_id].sort_values(ascending=False)
            similar_sessions = similar_sessions.drop(session_id, errors='ignore')
            
            similar_df = pd.DataFrame({
                "Session": similar_sessions.index,
                "Similarity": similar_sessions.values,
            })
            similar_df.insert(0, "Rank", range(1, len(similar_df) + 1))
            
            st.dataframe(
                similar_df.head(10),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Similarity": st.column_config.NumberColumn(format="%.3f"),
                },
            )


# ============================================================
# Main App
# ============================================================

def main():
    st.title("🔍 SMART Session Viewer")
    
    # Try to find bundle automatically
    default_bundle = find_bundle_path()
    
    # Sidebar for bundle selection (collapsed if default found)
    if default_bundle:
        # Default bundle found - show minimal UI
        with st.sidebar.expander("📁 Data Source", expanded=False):
            bundle_path = st.text_input(
                "Bundle Path",
                value=default_bundle,
                help="Path to viewer bundle directory or .zip file"
            )
    else:
        # No default - show prominent UI
        st.sidebar.header("📁 Data Source")
        bundle_path = st.sidebar.text_input(
            "Bundle Path",
            value="",
            help="Path to viewer bundle directory or .zip file"
        )
    
    if not bundle_path or not Path(bundle_path).exists():
        st.warning("Please specify a valid bundle path in the sidebar.")
        st.info(
            "A bundle should contain:\n"
            "- manifest.json\n"
            "- presentations.parquet\n"
            "- sessions.parquet\n"
            "- pres_similarities.parquet\n"
            "- session_similarities.parquet"
        )
        return
    
    # Load manifest
    manifest = load_bundle_manifest(bundle_path)
    
    if manifest:
        st.sidebar.markdown("---")
        st.sidebar.write(f"**Conference:** {manifest.get('conference_name', 'Unknown')}")
        st.sidebar.write(f"**Version:** {manifest.get('version_tag', 'Unknown')}")
        st.sidebar.write(f"**Presentations:** {manifest.get('presentation_count', '?')}")
        st.sidebar.write(f"**Sessions:** {manifest.get('session_count', '?')}")
    
    # Password for encrypted data
    has_encrypted = manifest.get("has_encrypted_data", False)
    password = None
    use_encrypted = False
    
    if has_encrypted:
        st.sidebar.markdown("---")
        st.sidebar.write("🔒 **Encrypted data available**")
        password = st.sidebar.text_input("Password", type="password")
        
        if password:
            # Validate password (you can add HMAC check here)
            use_encrypted = True
            st.sidebar.success("Password entered")
    
    # Load data
    try:
        df_presentations = load_presentations(bundle_path, use_encrypted, password)
        df_sessions = load_sessions(bundle_path)
        df_pres_sim = load_pres_similarities(bundle_path)
        df_sess_sim = load_session_similarities(bundle_path)
    except Exception as e:
        st.error(f"Error loading bundle: {e}")
        return
    
    # Set index for similarity lookups
    if "Abstract ID" in df_presentations.columns:
        df_presentations = df_presentations.set_index("Abstract ID", drop=False)
    
    # Determine session column
    session_col = "Session Code"
    if session_col not in df_presentations.columns:
        for col in ["session_id", "cluster_id"]:
            if col in df_presentations.columns:
                session_col = col
                break
    
    # Introduction
    st.markdown("""
    This tool allows you to explore the similarity between presentations and sessions.
    Similarity is based on text embeddings of titles and abstracts.
    """)
    
    show_metric_descriptions()
    
    # Tabs
    tab_pres, tab_sess = st.tabs(["📄 Presentations", "📁 Sessions"])
    
    with tab_pres:
        show_presentations_tab(df_presentations, df_pres_sim, session_col)
    
    with tab_sess:
        show_sessions_tab(df_sessions, df_presentations, df_sess_sim, session_col)


if __name__ == "__main__":
    main()

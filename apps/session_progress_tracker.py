"""
SMART Session Progress Tracker

A Streamlit application for tracking session organization progress through the year.
Calculates metrics for sessions that have already been created and assigned.

This tool is designed for organizers who:
- Already have sessions created and assigned
- Want to track how sessions are evolving week-to-week
- Need to generate viewer-compatible exports for team review

Usage:
    streamlit run session_progress_tracker.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

# Page config must be first
st.set_page_config(
    page_title="SMART Progress Tracker",
    page_icon=":material/trending_up:",
    layout="wide",
)

# Load environment
load_dotenv(".env")


# ============================================================
# Session State Management
# ============================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "step": 1,
        "df_presentations": None,
        "df_sessions": None,
        "embeddings": None,
        "abstract_ids": None,
        "session_col": None,
        "id_col": None,
        "title_col": None,
        "abstract_col": None,
        "metrics_calculated": False,
        "conference_name": "Conference",
        "embedding_cache_path": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# Step 1: Load Data
# ============================================================

def render_step1_load_data():
    """Render the data loading step."""
    st.header("Step 1: Load Presentation Data")
    
    st.markdown("""
    Upload a file containing presentations with **session assignments**.
    The file should have columns for:
    - Presentation ID
    - Title
    - Abstract (optional, for embedding)
    - **Session** (the assigned session name/code)
    """)
    
    # Conference name
    st.session_state.conference_name = st.text_input(
        "Conference Name",
        value=st.session_state.conference_name,
        help="Used for output file naming"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload presentations file",
        type=["csv", "xlsx", "parquet"],
        help="CSV, Excel, or Parquet file with session assignments"
    )
    
    if uploaded_file is not None:
        # Load file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Show preview
            with st.expander("Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Column mapping
            st.subheader("Map Columns")
            
            columns = [""] + list(df.columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Auto-detect common column names
                id_default = _find_column(df.columns, ["Abstract ID", "abstract_id", "Submission ID", "submission_id", "ID"])
                title_default = _find_column(df.columns, ["Title", "title", "Presentation Title"])
                
                id_col = st.selectbox("Presentation ID Column", columns, index=columns.index(id_default) if id_default else 0)
                title_col = st.selectbox("Title Column", columns, index=columns.index(title_default) if title_default else 0)
            
            with col2:
                abstract_default = _find_column(df.columns, ["Abstract", "abstract", "Description"])
                session_default = _find_column(df.columns, ["Session", "session", "Session Code", "session_id", "cluster_id"])
                
                abstract_col = st.selectbox("Abstract Column (optional)", columns, index=columns.index(abstract_default) if abstract_default else 0)
                session_col = st.selectbox("Session Column", columns, index=columns.index(session_default) if session_default else 0)
            
            # Validate
            if id_col and title_col and session_col:
                # Store in session state
                st.session_state.df_presentations = df
                st.session_state.id_col = id_col
                st.session_state.title_col = title_col
                st.session_state.abstract_col = abstract_col if abstract_col else None
                st.session_state.session_col = session_col
                
                # Show session summary
                session_counts = df[session_col].value_counts()
                
                st.subheader("Session Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Presentations", len(df))
                with col2:
                    st.metric("Total Sessions", len(session_counts))
                with col3:
                    st.metric("Avg Session Size", f"{len(df) / len(session_counts):.1f}")
                
                # Next step button
                if st.button("Continue to Embeddings →", type="primary"):
                    st.session_state.step = 2
                    st.rerun()
            else:
                st.warning("Please map all required columns (ID, Title, Session)")
        
        except Exception as e:
            st.error(f"Error loading file: {e}")


def _find_column(columns, candidates):
    """Find a column matching any of the candidate names."""
    for candidate in candidates:
        for col in columns:
            if col.lower() == candidate.lower():
                return col
    return None


# ============================================================
# Step 2: Generate/Load Embeddings
# ============================================================

def render_step2_embeddings():
    """Render the embeddings generation step."""
    st.header("Step 2: Generate Embeddings")
    
    df = st.session_state.df_presentations
    title_col = st.session_state.title_col
    abstract_col = st.session_state.abstract_col
    id_col = st.session_state.id_col
    
    # Create combined text for embedding
    if abstract_col and abstract_col in df.columns:
        texts = (df[title_col].fillna('') + " " + df[abstract_col].fillna('')).tolist()
        st.info(f"Using title + abstract for embeddings ({len(texts)} presentations)")
    else:
        texts = df[title_col].fillna('').tolist()
        st.info(f"Using title only for embeddings ({len(texts)} presentations)")
    
    abstract_ids = df[id_col].astype(str).tolist()
    
    # Embedding options
    st.subheader("Embedding Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        embedding_backend = st.selectbox(
            "Embedding Backend",
            ["Gemini", "Ollama (Local)"],
            help="Gemini requires API key, Ollama runs locally"
        )
        
        if embedding_backend == "Gemini":
            model = st.selectbox(
                "Model",
                ["gemini-embedding-001", "text-embedding-004"],
            )
            
            if "GEMINI_API_KEY" not in os.environ:
                api_key = st.text_input("Gemini API Key", type="password")
                if api_key:
                    os.environ["GEMINI_API_KEY"] = api_key
        else:
            model = st.text_input("Ollama Model", value="nomic-embed-text")
    
    with col2:
        # Cache configuration
        cache_path = st.text_input(
            "Embedding Cache Database",
            value=f"{st.session_state.conference_name}_embeddings.db",
            help="Cached embeddings from previous runs will be reused"
        )
        st.session_state.embedding_cache_path = cache_path
    
    # Check for existing cache
    cache_exists = Path(cache_path).exists()
    if cache_exists:
        try:
            from smart import EmbeddingCache
            cache = EmbeddingCache(cache_path)
            stats = cache.get_stats()
            st.success(f"✓ Found existing cache with {stats['total_embeddings']} embeddings")
        except Exception:
            pass
    
    # Generate button
    if st.button("Generate Embeddings", type="primary"):
        if embedding_backend == "Gemini" and "GEMINI_API_KEY" not in os.environ:
            st.error("Please provide Gemini API key")
            return
        
        with st.spinner("Generating embeddings..."):
            try:
                from smart import EmbeddingCache
                from smart.llm.embeddings import GeminiEmbedder, OllamaEmbedder, CachedEmbedder
                
                # Initialize cache
                cache = EmbeddingCache(cache_path)
                
                # Initialize embedder
                if embedding_backend == "Gemini":
                    base_embedder = GeminiEmbedder(model=model)
                else:
                    base_embedder = OllamaEmbedder(model=model)
                
                embedder = CachedEmbedder(base_embedder, cache)
                
                # Generate embeddings with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Check cache first
                cached_count = 0
                for text in texts:
                    if cache.get_embedding(text, embedder.config) is not None:
                        cached_count += 1
                
                status_text.text(f"Found {cached_count}/{len(texts)} in cache")
                
                # Generate embeddings
                embeddings_list = embedder.embed_batch(texts, show_progress=False)
                
                for i in range(len(texts)):
                    progress_bar.progress((i + 1) / len(texts))
                
                embeddings = np.array(embeddings_list)
                
                # Store in session state
                st.session_state.embeddings = embeddings
                st.session_state.abstract_ids = abstract_ids
                
                progress_bar.progress(1.0)
                status_text.text(f"✓ Generated {len(embeddings)} embeddings")
                
                st.success(f"Embeddings complete: shape {embeddings.shape}")
                
                # Auto-advance
                st.session_state.step = 3
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating embeddings: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Navigation
    st.markdown("---")
    if st.button("← Back to Data Loading"):
        st.session_state.step = 1
        st.rerun()


# ============================================================
# Step 3: Calculate Metrics
# ============================================================

def render_step3_metrics():
    """Render the metrics calculation step."""
    st.header("Step 3: Calculate Session Metrics")
    
    df = st.session_state.df_presentations
    embeddings = st.session_state.embeddings
    abstract_ids = st.session_state.abstract_ids
    session_col = st.session_state.session_col
    id_col = st.session_state.id_col
    
    st.info(f"Ready to calculate metrics for {len(df)} presentations in {df[session_col].nunique()} sessions")
    
    if st.button("Calculate Metrics", type="primary"):
        with st.spinner("Calculating session metrics..."):
            try:
                from smart import calculate_all_metrics
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Build session assignments dict
                session_assignments = {}
                for _, row in df.iterrows():
                    aid = str(row[id_col])
                    session_id = row[session_col]
                    if pd.notna(session_id):
                        session_assignments[aid] = session_id
                
                # Calculate metrics
                metrics = calculate_all_metrics(
                    embeddings,
                    abstract_ids,
                    session_assignments,
                )
                
                # Build sessions DataFrame
                sessions_data = []
                for session_id, sm in metrics["session_metrics"].items():
                    sessions_data.append({
                        "session_id": session_id,
                        "session_size": sm["size"],
                        "session_coherence": sm["coherence"],
                        "session_std_dev": sm.get("std_dev", 0),
                        "distinctiveness": sm.get("distinctiveness", 0),
                    })
                
                df_sessions = pd.DataFrame(sessions_data)
                df_sessions = df_sessions.sort_values("session_id").reset_index(drop=True)
                
                # Add presentation-level metrics
                df_with_metrics = df.copy()
                
                # Build index mapping
                id_to_idx = {str(aid): i for i, aid in enumerate(abstract_ids)}
                
                # Calculate similarity matrix
                sim_matrix = cosine_similarity(embeddings, embeddings)
                
                fit_scores = []
                raw_devs = []
                std_devs = []
                
                for _, row in df_with_metrics.iterrows():
                    aid = str(row[id_col])
                    session_id = row[session_col]
                    
                    if pd.isna(session_id) or aid not in id_to_idx:
                        fit_scores.append(np.nan)
                        raw_devs.append(np.nan)
                        std_devs.append(np.nan)
                        continue
                    
                    pres_idx = id_to_idx[aid]
                    
                    # Get session members
                    session_mask = df_with_metrics[session_col] == session_id
                    session_aids = df_with_metrics.loc[session_mask, id_col].astype(str).tolist()
                    session_indices = [id_to_idx[a] for a in session_aids if a in id_to_idx]
                    
                    if len(session_indices) < 2:
                        fit_scores.append(1.0)
                        raw_devs.append(0.0)
                        std_devs.append(0.0)
                        continue
                    
                    # Fit = mean similarity to other session members
                    other_indices = [i for i in session_indices if i != pres_idx]
                    fit = np.mean(sim_matrix[pres_idx, other_indices]) if other_indices else 1.0
                    
                    # Get session coherence
                    session_metrics = metrics["session_metrics"].get(session_id, {})
                    coherence = session_metrics.get("coherence", 1.0)
                    
                    # Calculate session std dev
                    all_fits = []
                    for idx in session_indices:
                        other = [i for i in session_indices if i != idx]
                        if other:
                            all_fits.append(np.mean(sim_matrix[idx, other]))
                    session_std = np.std(all_fits, ddof=1) if len(all_fits) > 1 else 0.0
                    
                    raw_dev = fit - coherence
                    std_dev = raw_dev / session_std if session_std > 0 else 0.0
                    
                    fit_scores.append(float(fit))
                    raw_devs.append(float(raw_dev))
                    std_devs.append(float(std_dev))
                
                df_with_metrics["Presentation Session Fit"] = fit_scores
                df_with_metrics["Presentation Raw Deviation"] = raw_devs
                df_with_metrics["Presentation Standardized Deviation"] = std_devs
                
                # Store results
                st.session_state.df_presentations = df_with_metrics
                st.session_state.df_sessions = df_sessions
                st.session_state.metrics_calculated = True
                
                st.success("✓ Metrics calculated!")
                
                # Show summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Coherence", f"{metrics['summary']['mean_coherence']:.3f}")
                with col2:
                    st.metric("Std Coherence", f"{metrics['summary']['std_coherence']:.3f}")
                with col3:
                    st.metric("Mean Fit", f"{metrics['summary']['mean_fit']:.3f}")
                
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show results if calculated
    if st.session_state.metrics_calculated:
        st.subheader("Session Metrics")
        st.dataframe(
            st.session_state.df_sessions,
            use_container_width=True,
            column_config={
                "session_coherence": st.column_config.NumberColumn(format="%.3f"),
                "session_std_dev": st.column_config.NumberColumn(format="%.3f"),
                "distinctiveness": st.column_config.NumberColumn(format="%.3f"),
            }
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Coherence Distribution")
            st.bar_chart(
                st.session_state.df_sessions.set_index("session_id")["session_coherence"]
            )
        
        with col2:
            st.subheader("Session Size Distribution")
            st.bar_chart(
                st.session_state.df_sessions.set_index("session_id")["session_size"]
            )
        
        # Next step
        if st.button("Continue to Export →", type="primary"):
            st.session_state.step = 4
            st.rerun()
    
    # Navigation
    st.markdown("---")
    if st.button("← Back to Embeddings"):
        st.session_state.step = 2
        st.rerun()


# ============================================================
# Step 4: Export Results
# ============================================================

def render_step4_export():
    """Render the export step."""
    st.header("Step 4: Export Results")
    
    df = st.session_state.df_presentations
    df_sessions = st.session_state.df_sessions
    embeddings = st.session_state.embeddings
    abstract_ids = st.session_state.abstract_ids
    id_col = st.session_state.id_col
    title_col = st.session_state.title_col
    session_col = st.session_state.session_col
    
    st.markdown("""
    Export results as a **Viewer Bundle** for use with the Session Viewer app.
    The bundle includes all data needed for team review.
    """)
    
    # Export options
    st.subheader("Export Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        output_dir = st.text_input(
            "Output Directory",
            value="output",
            help="Directory for output files"
        )
        
        include_abstracts = st.checkbox(
            "Include Abstracts in Public Export",
            value=False,
            help="Whether to include abstract text in the public viewer"
        )
    
    with col2:
        include_embeddings = st.checkbox(
            "Include Embeddings",
            value=True,
            help="Include embeddings for future recalculation"
        )
        
        encrypt_password = st.text_input(
            "Encryption Password (optional)",
            type="password",
            help="Password to encrypt sensitive data"
        )
    
    # Export button
    if st.button("Export Viewer Bundle", type="primary"):
        with st.spinner("Exporting..."):
            try:
                from smart.io.exporters import export_viewer_bundle
                
                # Prepare presentations DataFrame with standard columns
                df_export = df.copy()
                
                # Rename columns for viewer compatibility
                rename_map = {
                    id_col: "Abstract ID",
                    title_col: "Title",
                    session_col: "Session Code",
                }
                df_export = df_export.rename(columns=rename_map)
                
                # Create output directory
                output_path = Path(output_dir) / f"{st.session_state.conference_name}_bundle"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Export bundle
                manifest = export_viewer_bundle(
                    df_presentations=df_export,
                    df_sessions=df_sessions,
                    embeddings=embeddings,
                    abstract_ids=abstract_ids,
                    output_path=output_path,
                    conference_name=st.session_state.conference_name,
                    version_tag=datetime.now().strftime("%Y%m%d_%H%M%S"),
                    include_abstracts=include_abstracts,
                    include_embeddings=include_embeddings,
                    encrypt_password=encrypt_password if encrypt_password else None,
                )
                
                st.success(f"✓ Bundle exported to: {manifest['bundle_path']}")
                
                # Show manifest
                with st.expander("Export Details"):
                    st.json(manifest)
                
                # Instructions
                st.info(
                    f"To view results, run:\n\n"
                    f"```\nstreamlit run session_viewer_app.py -- --bundle {manifest['bundle_path']}\n```"
                )
                
            except Exception as e:
                st.error(f"Error exporting: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Also offer CSV export
    st.markdown("---")
    st.subheader("Alternative Exports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Sessions CSV"):
            csv = df_sessions.to_csv(index=False)
            st.download_button(
                "📥 sessions.csv",
                csv,
                f"{st.session_state.conference_name}_sessions.csv",
                "text/csv"
            )
    
    with col2:
        if st.button("Download Presentations CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 presentations.csv",
                csv,
                f"{st.session_state.conference_name}_presentations.csv",
                "text/csv"
            )
    
    # Navigation
    st.markdown("---")
    if st.button("← Back to Metrics"):
        st.session_state.step = 3
        st.rerun()


# ============================================================
# Main App
# ============================================================

def main():
    init_session_state()
    
    st.title("📊 SMART Session Progress Tracker")
    
    st.markdown("""
    Track session organization progress throughout the year. 
    Load presentations with existing session assignments, calculate metrics, 
    and export for team review.
    """)
    
    # Progress indicator
    steps = ["1. Load Data", "2. Embeddings", "3. Metrics", "4. Export"]
    current_step = st.session_state.step
    
    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps), 1):
        if i < current_step:
            col.success(f"✓ {step_name}")
        elif i == current_step:
            col.info(f"▶ {step_name}")
        else:
            col.write(f"○ {step_name}")
    
    st.markdown("---")
    
    # Render current step
    if current_step == 1:
        render_step1_load_data()
    elif current_step == 2:
        render_step2_embeddings()
    elif current_step == 3:
        render_step3_metrics()
    elif current_step == 4:
        render_step4_export()


if __name__ == "__main__":
    main()

"""
SMART - Session Matching And Automated Recommendation Tool

Streamlit-based local application for conference session organization.
"""

import os
import json
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import SMART modules
from smart.core.database import EmbeddingCache, ConferenceDB, EmbeddingConfig
from smart.io.loaders import (
    inspect_file, ColumnMapper, ColumnMapping, 
    load_presentations, load_hybrid_sessions, load_committees
)
from smart.io.exporters import (
    ExportProfile, PROFILE_CLOUD_VIEWER, PROFILE_ORGANIZER_FULL,
    PROFILE_ROOM_ASSIGNMENT, export_for_viewer, export_spreadsheet
)
from smart.llm.embeddings import create_embedder, CachedEmbedder
from smart.llm.titles import create_title_generator, CachedTitleGenerator
from smart.core.placement import create_placement_strategy, SessionConstraints
from smart.core.metrics import calculate_all_metrics, find_outlier_presentations

# App configuration
st.set_page_config(
    page_title="SMART Session Organizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Config file path
CONFIG_DIR = Path.home() / ".smart"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    """Load app configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {
        "default_backend": "gemini",
        "default_embedding_model": "gemini-embedding-001",
        "default_title_model": "gemini-2.0-flash",
        "recent_conferences": [],
        "ollama_host": "http://localhost:11434",
    }


def save_config(config: dict):
    """Save app configuration."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def init_session_state():
    """Initialize Streamlit session state."""
    defaults = {
        "current_step": 0,
        "conference_db": None,
        "embedding_cache": None,
        "embedder": None,
        "presentations_df": None,
        "embeddings": None,
        "abstract_ids": None,
        "sessions_created": False,
        "file_inspection": None,
        "column_mapping": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# === Wizard Steps ===

def step_conference_setup():
    """Step 1: Create or open conference."""
    st.header("📁 Conference Setup")
    
    config = load_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create New Conference")
        
        conference_name = st.text_input(
            "Conference Name",
            value=f"AIM{datetime.now().year % 100}",
            help="Short identifier for the conference"
        )
        
        conference_year = st.number_input(
            "Conference Year (2-digit)",
            min_value=20, max_value=99,
            value=datetime.now().year % 100,
        )
        
        working_dir = st.text_input(
            "Working Directory",
            value=str(Path.cwd()),
            help="Directory for database files"
        )
        
        if st.button("Create Conference", type="primary"):
            working_path = Path(working_dir)
            working_path.mkdir(parents=True, exist_ok=True)
            
            # Create databases
            cache_path = working_path / f"{conference_name}_cache.db"
            working_path_db = working_path / f"{conference_name}_working.db"
            
            st.session_state.embedding_cache = EmbeddingCache(cache_path)
            st.session_state.conference_db = ConferenceDB(
                working_path_db, conference_year=conference_year
            )
            
            # Update recent conferences
            recent = config.get("recent_conferences", [])
            entry = {
                "name": conference_name,
                "cache_path": str(cache_path),
                "working_path": str(working_path_db),
                "year": conference_year,
            }
            if entry not in recent:
                recent.insert(0, entry)
                recent = recent[:10]  # Keep last 10
            config["recent_conferences"] = recent
            save_config(config)
            
            st.success(f"Created conference: {conference_name}")
            st.session_state.current_step = 1
            st.rerun()
    
    with col2:
        st.subheader("Open Existing Conference")
        
        recent = config.get("recent_conferences", [])
        if recent:
            options = [f"{c['name']} ({c['year']})" for c in recent]
            selected = st.selectbox("Recent Conferences", options)
            
            if selected and st.button("Open Conference"):
                idx = options.index(selected)
                conf = recent[idx]
                
                st.session_state.embedding_cache = EmbeddingCache(conf["cache_path"])
                st.session_state.conference_db = ConferenceDB(
                    conf["working_path"], conference_year=conf["year"]
                )
                
                st.success(f"Opened conference: {conf['name']}")
                st.session_state.current_step = 1
                st.rerun()
        else:
            st.info("No recent conferences found")
        
        st.divider()
        
        # Manual file selection
        cache_file = st.file_uploader(
            "Or select cache database (.db)",
            type=["db"],
            key="cache_upload"
        )
        working_file = st.file_uploader(
            "And working database (.db)",
            type=["db"],
            key="working_upload"
        )


def step_import_data():
    """Step 2: Import presentation data."""
    st.header("📥 Import Data")
    
    if st.session_state.conference_db is None:
        st.warning("Please create or open a conference first")
        return
    
    # Show current stats
    stats = st.session_state.conference_db.get_stats()
    st.info(f"Current: {stats['total_presentations']} presentations, {stats['total_sessions']} sessions")
    
    # Option to clear existing presentations for reload
    if stats['total_presentations'] > 0:
        with st.expander("⚠️ Clear Existing Presentations (for reload)"):
            st.write("""
            Use this to replace the current presentation list with a new one.
            **Embeddings are preserved** in the cache and will be reused for matching texts.
            """)
            col_clear1, col_clear2 = st.columns(2)
            with col_clear1:
                if st.button("Clear Presentations Only", type="secondary"):
                    count = st.session_state.conference_db.clear_presentations(keep_sessions=False)
                    st.success(f"Cleared {count} presentations. You can now import a new set.")
                    st.rerun()
            with col_clear2:
                if st.button("Clear Presentations & Sessions", type="secondary"):
                    count = st.session_state.conference_db.clear_presentations(keep_sessions=False)
                    st.success(f"Cleared {count} presentations and all sessions.")
                    st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["📄 Regular Presentations", "⭐ Hybrid Sessions", "🏛️ Committees"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload presentations file (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="pres_upload"
        )
        
        if uploaded_file:
            # Save temporarily and inspect
            temp_path = Path(f"/tmp/{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            inspection = inspect_file(temp_path)
            st.session_state.file_inspection = inspection
            
            st.write(f"**{inspection['row_count']} rows, {inspection['column_count']} columns**")
            
            # Show preview of first 5 rows
            st.subheader("Data Preview (First 5 Rows)")
            if temp_path.suffix.lower() == '.csv':
                preview_df = pd.read_csv(temp_path, nrows=5)
            else:
                preview_df = pd.read_excel(temp_path, nrows=5)
            st.dataframe(preview_df, width='stretch', height=200)
            
            # Show auto-detected mappings
            st.subheader("Column Mapping")
            auto_mapped = inspection["auto_detected_mappings"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                title_col = st.selectbox(
                    "Title Column *",
                    options=[""] + [c["name"] for c in inspection["columns"]],
                    index=([c["name"] for c in inspection["columns"]].index(auto_mapped.get("title", "")) + 1) 
                          if auto_mapped.get("title") in [c["name"] for c in inspection["columns"]] else 0
                )
                
                abstract_col = st.selectbox(
                    "Abstract Column",
                    options=["(none)"] + [c["name"] for c in inspection["columns"]],
                    index=([c["name"] for c in inspection["columns"]].index(auto_mapped.get("abstract", "")) + 1) 
                          if auto_mapped.get("abstract") in [c["name"] for c in inspection["columns"]] else 0
                )
                
                submission_id_col = st.selectbox(
                    "Abstract ID Column",
                    options=["(none)"] + [c["name"] for c in inspection["columns"]],
                    index=([c["name"] for c in inspection["columns"]].index(auto_mapped.get("submission_id", "")) + 1) 
                          if auto_mapped.get("submission_id") in [c["name"] for c in inspection["columns"]] else 0,
                    help="The Abstract ID (e.g., 26XXXXX) used to identify presentations. Often labeled 'Submission ID' in submission portals."
                )
            
            with col2:
                email_col = st.selectbox(
                    "Presenter Email Column",
                    options=["(none)"] + [c["name"] for c in inspection["columns"]],
                    index=([c["name"] for c in inspection["columns"]].index(auto_mapped.get("presenter_email", "")) + 1) 
                          if auto_mapped.get("presenter_email") in [c["name"] for c in inspection["columns"]] else 0
                )
                
                first_name_col = st.selectbox(
                    "Presenter First Name",
                    options=["(none)"] + [c["name"] for c in inspection["columns"]],
                    index=([c["name"] for c in inspection["columns"]].index(auto_mapped.get("presenter_first_name", "")) + 1) 
                          if auto_mapped.get("presenter_first_name") in [c["name"] for c in inspection["columns"]] else 0
                )
                
                last_name_col = st.selectbox(
                    "Presenter Last Name",
                    options=["(none)"] + [c["name"] for c in inspection["columns"]],
                    index=([c["name"] for c in inspection["columns"]].index(auto_mapped.get("presenter_last_name", "")) + 1) 
                          if auto_mapped.get("presenter_last_name") in [c["name"] for c in inspection["columns"]] else 0
                )
            
            if st.button("Import Presentations", type="primary", disabled=not title_col):
                try:
                    with st.spinner("Importing presentations..."):
                        # Create mapping
                        mapping = ColumnMapping(
                            title=title_col,
                            abstract=abstract_col if abstract_col != "(none)" else None,
                            submission_id=submission_id_col if submission_id_col != "(none)" else None,
                            presenter_email=email_col if email_col != "(none)" else None,
                            presenter_first_name=first_name_col if first_name_col != "(none)" else None,
                            presenter_last_name=last_name_col if last_name_col != "(none)" else None,
                        )
                        
                        # Load presentations
                        df, metadata = load_presentations(
                            temp_path, mapping,
                            conference_year=st.session_state.conference_db.conference_year
                        )
                        
                        # Import to database
                        presentations = df.to_dict('records')
                        ids = st.session_state.conference_db.import_presentations_batch(presentations)
                        
                        st.success(f"✓ Imported {len(ids)} presentations ({metadata['temp_id_count']} with temp IDs)")
                        st.session_state.presentations_df = df
                        st.rerun()
                except Exception as e:
                    st.error(f"Error importing presentations: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with tab2:
        st.write("Import pre-assigned hybrid sessions with invited presentations")
        st.caption("Hybrid sessions are pre-organized sessions (e.g., invited talks, special sessions) where presentations are already assigned to specific sessions.")
        
        hybrid_file = st.file_uploader(
            "Upload hybrid sessions file",
            type=["csv", "xlsx", "xls"],
            key="hybrid_upload"
        )
        
        if hybrid_file is not None:
            # Save to temp file for processing
            temp_path = Path(f"/tmp/{hybrid_file.name}")
            with open(temp_path, "wb") as f:
                f.write(hybrid_file.getvalue())
            
            try:
                inspection = inspect_file(temp_path)
                
                st.subheader("File Preview")
                st.caption(f"{inspection['row_count']} rows, {len(inspection['columns'])} columns")
                
                # Auto-mapping suggestions
                auto_mapped = inspection.get("auto_mapping", {})
                
                st.subheader("Column Mapping")
                st.write("Map your file columns. Session column is required to group presentations into hybrid sessions.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    hybrid_title_col = st.selectbox(
                        "Title Column *",
                        options=[""] + [c["name"] for c in inspection["columns"]],
                        index=([c["name"] for c in inspection["columns"]].index(auto_mapped.get("title", "")) + 1) 
                              if auto_mapped.get("title") in [c["name"] for c in inspection["columns"]] else 0,
                        key="hybrid_title"
                    )
                    
                    hybrid_abstract_col = st.selectbox(
                        "Abstract Column",
                        options=["(none)"] + [c["name"] for c in inspection["columns"]],
                        index=([c["name"] for c in inspection["columns"]].index(auto_mapped.get("abstract", "")) + 1) 
                              if auto_mapped.get("abstract") in [c["name"] for c in inspection["columns"]] else 0,
                        key="hybrid_abstract"
                    )
                    
                    hybrid_id_col = st.selectbox(
                        "Abstract ID Column",
                        options=["(none)"] + [c["name"] for c in inspection["columns"]],
                        index=([c["name"] for c in inspection["columns"]].index(auto_mapped.get("submission_id", "")) + 1) 
                              if auto_mapped.get("submission_id") in [c["name"] for c in inspection["columns"]] else 0,
                        key="hybrid_id"
                    )
                
                with col2:
                    hybrid_session_col = st.selectbox(
                        "Session/Group Column * (required for hybrid)",
                        options=[""] + [c["name"] for c in inspection["columns"]],
                        index=0,
                        key="hybrid_session",
                        help="Column that identifies which session each presentation belongs to"
                    )
                    
                    hybrid_email_col = st.selectbox(
                        "Presenter Email Column",
                        options=["(none)"] + [c["name"] for c in inspection["columns"]],
                        index=([c["name"] for c in inspection["columns"]].index(auto_mapped.get("presenter_email", "")) + 1) 
                              if auto_mapped.get("presenter_email") in [c["name"] for c in inspection["columns"]] else 0,
                        key="hybrid_email"
                    )
                
                if st.button("Import Hybrid Sessions", type="primary", disabled=not (hybrid_title_col and hybrid_session_col)):
                    try:
                        with st.spinner("Importing hybrid sessions..."):
                            # Create mapping with session column
                            mapping = ColumnMapping(
                                title=hybrid_title_col,
                                abstract=hybrid_abstract_col if hybrid_abstract_col != "(none)" else None,
                                submission_id=hybrid_id_col if hybrid_id_col != "(none)" else None,
                                presenter_email=hybrid_email_col if hybrid_email_col != "(none)" else None,
                                session=hybrid_session_col,
                            )
                            
                            # Load hybrid sessions
                            df_pres, df_sessions, metadata = load_hybrid_sessions(
                                temp_path, mapping,
                                conference_year=st.session_state.conference_db.conference_year
                            )
                            
                            # Import presentations to database
                            presentations = df_pres.to_dict('records')
                            st.session_state.conference_db.import_presentations_batch(presentations)
                            
                            # Create hybrid sessions with their assigned presentations
                            for _, session_row in df_sessions.iterrows():
                                st.session_state.conference_db.create_session(
                                    session_id=session_row["session_id"],
                                    title=session_row["title"],
                                    presentation_ids=session_row["presentation_ids"],
                                    is_hybrid=True,
                                    placement_strategy="hybrid_import",
                                )
                            
                            # Show success with multi-slot info if applicable
                            msg = f"✓ Imported {metadata['total_presentation_slots']} presentation slots in {len(df_sessions)} hybrid sessions"
                            if metadata.get("multi_slot_presentations", 0) > 0:
                                msg += f"\n  ({metadata['unique_presentations']} unique presentations, {metadata['multi_slot_presentations']} spanning multiple slots)"
                            st.success(msg)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error importing hybrid sessions: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with tab3:
        st.write("Import committee data for session assignment")
        st.caption("Committees can be matched to sessions based on topic similarity. Each committee needs a name and description.")
        
        committee_file = st.file_uploader(
            "Upload committee file",
            type=["csv", "xlsx", "xls"],
            key="committee_upload"
        )
        
        if committee_file is not None:
            # Save to temp file for processing
            temp_path = Path(f"/tmp/{committee_file.name}")
            with open(temp_path, "wb") as f:
                f.write(committee_file.getvalue())
            
            try:
                inspection = inspect_file(temp_path)
                
                st.subheader("File Preview")
                st.caption(f"{inspection['row_count']} rows, {len(inspection['columns'])} columns")
                
                st.subheader("Column Mapping")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    committee_name_col = st.selectbox(
                        "Committee Name Column *",
                        options=[""] + [c["name"] for c in inspection["columns"]],
                        index=0,
                        key="committee_name"
                    )
                
                with col2:
                    committee_desc_col = st.selectbox(
                        "Description Column *",
                        options=[""] + [c["name"] for c in inspection["columns"]],
                        index=0,
                        key="committee_desc"
                    )
                
                if st.button("Import Committees", type="primary", disabled=not (committee_name_col and committee_desc_col)):
                    try:
                        with st.spinner("Importing committees..."):
                            # Load committees
                            df_committees, metadata = load_committees(
                                temp_path,
                                name_column=committee_name_col,
                                description_column=committee_desc_col,
                            )
                            
                            # Import to database using proper API method
                            committees_data = df_committees.to_dict('records')
                            imported_ids = st.session_state.conference_db.import_committees_batch(committees_data)
                            
                            st.success(f"✓ Imported {len(imported_ids)} committees")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error importing committees: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Navigation
    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.current_step = 0
            st.rerun()
    with col2:
        if stats['total_presentations'] > 0:
            if st.button("Next: Embeddings →", type="primary"):
                st.session_state.current_step = 2
                st.rerun()


def step_embeddings():
    """Step 3: Configure and generate embeddings."""
    st.header("🧠 Generate Embeddings")
    
    if st.session_state.conference_db is None:
        st.warning("Please create or open a conference first")
        return
    
    config = load_config()
    
    # Get presentations
    presentations = st.session_state.conference_db.get_presentations()
    
    if not presentations:
        st.warning("No presentations loaded. Please import data first.")
        return
    
    # Backend selection
    col1, col2 = st.columns(2)
    
    with col1:
        backend = st.selectbox(
            "Embedding Backend",
            options=["gemini", "ollama", "sentence-transformers"],
            index=["gemini", "ollama", "sentence-transformers"].index(
                config.get("default_backend", "gemini")
            )
        )
        
        if backend == "gemini":
            model = st.text_input(
                "Model Name",
                value=config.get("default_embedding_model", "gemini-embedding-001")
            )
            
            api_key = st.text_input(
                "API Key (or set GEMINI_API_KEY env var)",
                type="password",
                value=""
            )
            
        elif backend == "ollama":
            model = st.text_input(
                "Model Name",
                value="nomic-embed-text"
            )
            
            host = st.text_input(
                "Ollama Host",
                value=config.get("ollama_host", "http://localhost:11434")
            )
        else:
            model = st.text_input(
                "Model Name",
                value="all-MiniLM-L6-v2"
            )
    
    with col2:
        task_type = st.selectbox(
            "Task Type",
            options=["SEMANTIC_SIMILARITY", "RETRIEVAL_DOCUMENT", "CLASSIFICATION"],
            index=0
        )
    
    # Check cache status for current settings
    st.divider()
    st.subheader("Cache Status")
    
    texts = [p["combined_text"] for p in presentations]
    abstract_ids = [p["abstract_id"] for p in presentations]
    
    # Check how many are already cached
    from smart.core.database import EmbeddingConfig
    embedding_config = EmbeddingConfig(
        model_name=model if model else "unknown",
        model_version=model if model else "unknown",
        task_type=task_type,
    )
    
    cached_count = 0
    uncached_count = 0
    cached_embeddings = {}
    
    # Only check cache if we have an embedding_cache initialized
    if st.session_state.embedding_cache is not None:
        try:
            # Check each text individually to find what's cached
            for i, text in enumerate(texts):
                cached_emb = st.session_state.embedding_cache.get_embedding(text, embedding_config)
                if cached_emb is not None:
                    cached_count += 1
                    cached_embeddings[text] = cached_emb
                else:
                    uncached_count += 1
        except Exception as e:
            st.warning(f"Error checking cache: {e}. Assuming all need generation.")
            uncached_count = len(texts)
    else:
        st.warning("Cache not initialized. Please create or open a conference first.")
        uncached_count = len(texts)
    
    # Display cache status metrics
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Presentations", len(presentations))
    with col_stat2:
        st.metric("Already Cached", cached_count, help="Embeddings already in cache for this model")
    with col_stat3:
        st.metric("Need Generation", uncached_count, help="Presentations requiring API call")
    
    # Show progress bar of cache coverage
    if len(presentations) > 0:
        cache_pct = cached_count / len(presentations) * 100
        st.progress(cache_pct / 100, text=f"{cache_pct:.1f}% cached")
    
    st.divider()
    
    # Generation buttons
    col_gen, col_load = st.columns(2)
    
    with col_gen:
        generate_label = "Generate Embeddings" if uncached_count == len(presentations) else f"Generate {uncached_count} Missing Embeddings"
        generate_disabled = uncached_count == 0
        
        if st.button(generate_label, type="primary", disabled=generate_disabled):
            try:
                # Create embedder
                kwargs = {}
                if backend == "gemini":
                    # Use API key from input, env var, or .env file
                    actual_api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
                    if actual_api_key:
                        kwargs["api_key"] = actual_api_key
                elif backend == "ollama":
                    kwargs["host"] = host
                
                embedder = create_embedder(
                    backend=backend,
                    model=model,
                    cache=st.session_state.embedding_cache,
                    task_type=task_type,
                    **kwargs
                )
                st.session_state.embedder = embedder
                
                # Generate embeddings with progress (cache-aware)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_embeddings = []
                newly_generated = 0
                from_cache = 0
                batch_size = 10
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = embedder.embed_batch(batch_texts)
                    
                    for text, emb in zip(batch_texts, batch_embeddings):
                        all_embeddings.append(emb)
                        if text in cached_embeddings:
                            from_cache += 1
                        else:
                            newly_generated += 1
                    
                    progress = (i + len(batch_texts)) / len(texts)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {i + len(batch_texts)}/{len(texts)} ({from_cache} cached, {newly_generated} new)")
                
                st.session_state.embeddings = np.array(all_embeddings)
                st.session_state.abstract_ids = abstract_ids
                
                # Update database with embedding hashes
                for aid, text in zip(abstract_ids, texts):
                    import hashlib
                    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                    st.session_state.conference_db.update_embedding_hash(aid, text_hash)
                
                st.success(f"✓ {newly_generated} embeddings generated, {from_cache} loaded from cache")
                
                # Save config
                config["default_backend"] = backend
                config["default_embedding_model"] = model
                save_config(config)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating embeddings: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    with col_load:
        load_disabled = cached_count == 0
        
        if st.button("Load All from Cache", disabled=load_disabled, 
                     help="Load all cached embeddings without calling API"):
            if cached_count < len(presentations):
                st.warning(f"Only {cached_count}/{len(presentations)} presentations are cached. "
                          f"Use 'Generate Missing Embeddings' to compute the remaining {uncached_count}.")
            else:
                # All embeddings are cached - load them
                embeddings = []
                for text in texts:
                    embeddings.append(cached_embeddings[text])
                
                st.session_state.embeddings = np.array(embeddings)
                st.session_state.abstract_ids = abstract_ids
                st.success(f"✓ Loaded {len(embeddings)} embeddings from cache!")
                st.rerun()
    
    # Show current embeddings status
    st.divider()
    if st.session_state.embeddings is not None:
        emb_count = len(st.session_state.embeddings)
        pres_count = len(presentations)
        if emb_count == pres_count:
            st.success(f"✓ {emb_count}/{pres_count} presentations have embeddings and are ready for processing")
        else:
            st.warning(f"⚠ {emb_count}/{pres_count} presentations have embeddings")
    else:
        st.info(f"0/{len(presentations)} presentations have embeddings loaded")
    
    # Navigation
    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.current_step = 1
            st.rerun()
    with col2:
        if st.session_state.embeddings is not None:
            if st.button("Next: Check Duplicates →", type="primary"):
                st.session_state.current_step = 3
                st.rerun()


def step_check_duplicates():
    """Step 4: Check for duplicate presentations using embedding similarity."""
    st.header("🔍 Check for Duplicates")
    
    if st.session_state.embeddings is None:
        st.warning("Please generate embeddings first")
        return
    
    presentations = st.session_state.conference_db.get_presentations()
    embeddings = st.session_state.embeddings
    abstract_ids = st.session_state.abstract_ids
    
    st.write("""
    Identify potential duplicate submissions using embedding similarity.
    Near-duplicates (resubmissions with minor changes) are detected by comparing 
    the semantic similarity of presentation texts.
    
    **Note:** Invited presentations from hybrid sessions are marked with `-INV` in their ID 
    and can be identified by this suffix.
    """)
    
    # Count invited presentations
    invited_count = sum(1 for aid in abstract_ids if '-INV' in aid)
    if invited_count > 0:
        st.info(f"📌 {invited_count} invited presentations (marked with -INV) are included in the list")
    
    # Similarity threshold slider
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.80, max_value=1.00, value=0.95, step=0.01,
        help="Presentations with similarity above this threshold are flagged as potential duplicates. "
             "Higher values = stricter matching (fewer false positives)."
    )
    
    # Calculate pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    
    if len(embeddings) > 1:
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find duplicates above threshold (excluding self-similarity)
        duplicates_found = []
        seen_pairs = set()
        
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                sim = similarity_matrix[i, j]
                if sim >= similarity_threshold:
                    pair_key = tuple(sorted([abstract_ids[i], abstract_ids[j]]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        # Check if either is an invited presentation
                        is_inv_1 = '-INV' in abstract_ids[i]
                        is_inv_2 = '-INV' in abstract_ids[j]
                        duplicates_found.append({
                            "id_1": abstract_ids[i],
                            "id_2": abstract_ids[j],
                            "similarity": sim,
                            "title_1": presentations[i]["title"],
                            "title_2": presentations[j]["title"],
                            "idx_1": i,
                            "idx_2": j,
                            "is_invited_1": is_inv_1,
                            "is_invited_2": is_inv_2,
                        })
        
        # Display results
        if len(duplicates_found) == 0:
            st.success(f"✓ No duplicates found at {similarity_threshold:.0%} similarity threshold")
        else:
            st.warning(f"Found {len(duplicates_found)} potential duplicate pairs")
            
            # Initialize removal tracking
            if 'ids_to_remove' not in st.session_state:
                st.session_state.ids_to_remove = set()
            
            # Create dataframe for display
            df_dups = pd.DataFrame(duplicates_found)
            df_dups = df_dups.sort_values('similarity', ascending=False)
            
            # Display duplicates with selection
            for idx, row in df_dups.iterrows():
                # Add invited indicator to header
                inv_marker_1 = " ⭐INV" if row.get('is_invited_1') else ""
                inv_marker_2 = " ⭐INV" if row.get('is_invited_2') else ""
                
                with st.expander(f"**{row['similarity']:.1%}** similarity: {row['id_1']}{inv_marker_1} ↔ {row['id_2']}{inv_marker_2}"):
                    # Create two-column layout for comparison
                    col1, col2 = st.columns(2)
                    
                    pres1 = next((p for p in presentations if p['abstract_id'] == row['id_1']), {})
                    pres2 = next((p for p in presentations if p['abstract_id'] == row['id_2']), {})
                    
                    with col1:
                        header_1 = f"### {row['id_1']}"
                        if row.get('is_invited_1'):
                            header_1 += " ⭐"
                            st.markdown(header_1)
                            st.caption("**Invited Presentation** - Consider keeping this one")
                        else:
                            st.markdown(header_1)
                        st.write(f"**Title:** {row['title_1']}")
                        if pres1.get('abstract'):
                            st.write(f"**Abstract:** {pres1['abstract']}")
                        
                        # Show full details in nested expander
                        with st.expander(f"📋 All Details for {row['id_1']}", expanded=False):
                            # Convert all values to strings to avoid Arrow serialization issues
                            detail_data = {k: str(v) if v is not None else "" for k, v in pres1.items()}
                            detail_df = pd.DataFrame.from_dict(detail_data, orient='index', columns=['Value'])
                            st.dataframe(detail_df, width='stretch')
                        
                        remove_1 = st.checkbox(
                            f"Remove {row['id_1']}", 
                            value=row['id_1'] in st.session_state.ids_to_remove,
                            key=f"rm_{row['id_1']}_{row['id_2']}",
                            help="⚠️ This is an invited presentation" if row.get('is_invited_1') else None
                        )
                        if remove_1:
                            st.session_state.ids_to_remove.add(row['id_1'])
                        elif row['id_1'] in st.session_state.ids_to_remove:
                            st.session_state.ids_to_remove.discard(row['id_1'])
                    
                    with col2:
                        header_2 = f"### {row['id_2']}"
                        if row.get('is_invited_2'):
                            header_2 += " ⭐"
                            st.markdown(header_2)
                            st.caption("**Invited Presentation** - Consider keeping this one")
                        else:
                            st.markdown(header_2)
                        st.write(f"**Title:** {row['title_2']}")
                        if pres2.get('abstract'):
                            st.write(f"**Abstract:** {pres2['abstract']}")
                        
                        # Show full details in nested expander
                        with st.expander(f"📋 All Details for {row['id_2']}", expanded=False):
                            # Convert all values to strings to avoid Arrow serialization issues
                            detail_data = {k: str(v) if v is not None else "" for k, v in pres2.items()}
                            detail_df = pd.DataFrame.from_dict(detail_data, orient='index', columns=['Value'])
                            st.dataframe(detail_df, width='stretch')
                        
                        remove_2 = st.checkbox(
                            f"Remove {row['id_2']}", 
                            value=row['id_2'] in st.session_state.ids_to_remove,
                            key=f"rm_{row['id_2']}_{row['id_1']}",
                            help="⚠️ This is an invited presentation" if row.get('is_invited_2') else None
                        )
                        if remove_2:
                            st.session_state.ids_to_remove.add(row['id_2'])
                        elif row['id_2'] in st.session_state.ids_to_remove:
                            st.session_state.ids_to_remove.discard(row['id_2'])
                
            st.divider()
            
            # Summary and actions
            if st.session_state.ids_to_remove:
                st.info(f"**{len(st.session_state.ids_to_remove)}** presentations marked for removal: "
                       f"{', '.join(sorted(st.session_state.ids_to_remove))}")
                
                col_act1, col_act2, col_act3 = st.columns(3)
                
                with col_act1:
                    if st.button("🗑️ Remove Selected", type="primary"):
                        import sqlite3
                        ids_list = list(st.session_state.ids_to_remove)
                        with sqlite3.connect(st.session_state.conference_db.db_path) as conn:
                            placeholders = ','.join('?' * len(ids_list))
                            conn.execute(f"DELETE FROM presentations WHERE abstract_id IN ({placeholders})", ids_list)
                            conn.commit()
                        
                        # Update embeddings array
                        keep_mask = [aid not in st.session_state.ids_to_remove for aid in abstract_ids]
                        st.session_state.embeddings = embeddings[keep_mask]
                        st.session_state.abstract_ids = [aid for aid in abstract_ids if aid not in st.session_state.ids_to_remove]
                        
                        st.success(f"Removed {len(ids_list)} presentations")
                        st.session_state.ids_to_remove = set()
                        st.rerun()
                
                with col_act2:
                    # Download removed list
                    removed_df = pd.DataFrame([
                        p for p in presentations if p['abstract_id'] in st.session_state.ids_to_remove
                    ])
                    if len(removed_df) > 0:
                        csv_data = removed_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Removed List",
                            csv_data,
                            file_name="duplicates_to_remove.csv",
                            mime="text/csv"
                        )
                
                with col_act3:
                    if st.button("Clear Selection"):
                        st.session_state.ids_to_remove = set()
                        st.rerun()
    else:
        st.info("Need at least 2 presentations to check for duplicates")
    
    # Navigation
    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.current_step = 2
            st.rerun()
    with col2:
        if st.button("Next: Create Sessions →", type="primary"):
            st.session_state.current_step = 4
            st.rerun()


def step_create_sessions():
    """Step 5: Create sessions using placement strategy."""
    st.header("📊 Create Sessions")
    
    if st.session_state.embeddings is None:
        st.warning("Please generate embeddings first")
        return
    
    # Strategy selection
    col1, col2 = st.columns(2)
    
    STRATEGY_LABELS = {
        "oral": "Bottom-up Hierarchical (No Hybrid Fill)",
        "hybrid_first": "Bottom-up Hierarchical (Hybrid First) - Recommended",
        "traditional": "Traditional fcluster (for comparison)",
        "poster_thematic": "Poster Thematic Ordering",
    }
    
    with col1:
        strategy_type = st.selectbox(
            "Placement Strategy",
            options=["hybrid_first", "oral", "traditional", "poster_thematic"],
            format_func=lambda x: STRATEGY_LABELS.get(x, x),
        )
        
        st.caption(
            "**Hybrid First**: Fills hybrid/invited sessions with similar presentations first, "
            "then clusters remaining presentations. "
            "**No Hybrid Fill**: Just excludes pre-assigned presentations from clustering. "
            "**Traditional fcluster**: Cuts the tree at a fixed level."
        )
        
        min_size = st.number_input(
            "Minimum Session Size",
            min_value=4, max_value=20, value=8
        )
        
        max_size = st.number_input(
            "Maximum Session Size",
            min_value=8, max_value=30, value=12
        )
    
    with col2:
        max_sessions = st.number_input(
            "Maximum Sessions (0 = auto)",
            min_value=0, max_value=200, value=0
        )
        
        linkage_method = st.selectbox(
            "Linkage Method",
            options=["average", "complete", "single", "ward"],
            index=0
        )
        
        # Tree merge stop - expressed as "exclude bottom X%"
        exclude_bottom_pct = st.slider(
            "Exclude Bottom % of Relationships",
            min_value=0, max_value=20, value=5,
            help="Exclude the least related presentations from forming new sessions. "
                 "These will be assigned to existing sessions instead. "
                 "Higher values = stricter session formation, fewer weak sessions."
        )
        tree_merge_stop = 1.0 - (exclude_bottom_pct / 100.0)
    
    if st.button("Create Sessions", type="primary"):
        try:
            constraints = SessionConstraints(
                min_session_size=min_size,
                max_session_size=max_size,
                max_sessions=max_sessions if max_sessions > 0 else None,
            )
            
            strategy = create_placement_strategy(
                strategy_type=strategy_type,
                linkage_method=linkage_method,
                tree_merge_stop=tree_merge_stop,
            )
            
            # Get hybrid assignments from database (sessions marked as hybrid)
            hybrid_assignments = {}
            hybrid_embeddings = {}
            
            sessions = st.session_state.conference_db.get_sessions()
            for session in sessions:
                if session.get("is_hybrid"):
                    session_detail = st.session_state.conference_db.get_session(session["session_id"])
                    if session_detail and "presentations" in session_detail:
                        for pres in session_detail["presentations"]:
                            aid = pres["abstract_id"]
                            hybrid_assignments[aid] = session["session_id"]
                            # Get embedding if available
                            if aid in st.session_state.abstract_ids:
                                idx = st.session_state.abstract_ids.index(aid)
                                hybrid_embeddings[aid] = st.session_state.embeddings[idx]
            
            if hybrid_assignments:
                st.info(f"Found {len(hybrid_assignments)} presentations in {len(set(hybrid_assignments.values()))} hybrid sessions")
            
            # Pass hybrid_embeddings only for hybrid_first strategy
            place_kwargs = {
                "embeddings": st.session_state.embeddings,
                "abstract_ids": st.session_state.abstract_ids,
                "constraints": constraints,
                "hybrid_assignments": hybrid_assignments,
            }
            
            # HybridFirstPlacement accepts hybrid_embeddings
            if strategy_type == "hybrid_first" and hybrid_embeddings:
                place_kwargs["hybrid_embeddings"] = hybrid_embeddings
            
            result = strategy.place(**place_kwargs)
            
            # Save sessions to database
            for session in result.sessions:
                st.session_state.conference_db.create_session(
                    session_id=session["session_id"],
                    presentation_ids=session["presentation_ids"],
                    is_hybrid=session.get("is_hybrid", False),
                    placement_strategy=strategy.name,
                )
            
            st.session_state.sessions_created = True
            
            # Show summary - handle different metadata key names across strategies
            n_sessions = result.metadata.get('n_sessions') or result.metadata.get('n_total_sessions') or len(result.sessions)
            st.success(f"Created {n_sessions} sessions")
            st.write(f"- Assigned: {result.metadata.get('n_assigned', len(result.session_assignments))} presentations")
            st.write(f"- Unassigned: {result.metadata.get('n_unassigned', len(result.unassigned))} presentations")
            
            # Calculate metrics
            metrics = calculate_all_metrics(
                st.session_state.embeddings,
                st.session_state.abstract_ids,
                result.session_assignments,
            )
            
            st.metric("Mean Coherence", f"{metrics['summary']['mean_coherence']:.3f}")
            st.metric("Mean Fit", f"{metrics['summary']['mean_fit']:.3f}")
            
            # Update session metrics in database
            for session_id, sm in metrics["session_metrics"].items():
                st.session_state.conference_db.update_session_metrics(
                    session_id, sm["coherence"], sm.get("distinctiveness")
                )
            
        except Exception as e:
            st.error(f"Error creating sessions: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Navigation
    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.current_step = 3
            st.rerun()
    with col2:
        if st.session_state.sessions_created:
            if st.button("Next: Generate Titles →", type="primary"):
                st.session_state.current_step = 5
                st.rerun()


def step_generate_titles():
    """Step 6: Generate session titles and keywords."""
    st.header("✨ Generate Session Titles")
    
    if not st.session_state.sessions_created:
        st.warning("Please create sessions first")
        return
    
    config = load_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        backend = st.selectbox(
            "Title Generation Backend",
            options=["gemini", "ollama"],
            index=0
        )
        
        if backend == "gemini":
            model = st.text_input(
                "Model Name",
                value=config.get("default_title_model", "gemini-2.0-flash")
            )
            
            api_key = st.text_input(
                "Gemini API Key (or set GEMINI_API_KEY env var)",
                type="password",
                value=""
            )
        else:
            model = st.text_input(
                "Model Name",
                value="llama3.2:3b"
            )
            
            ollama_host = st.text_input(
                "Ollama Host",
                value=config.get("ollama_host", "http://localhost:11434")
            )
    
    with col2:
        force_regenerate = st.checkbox(
            "Force regenerate all titles",
            value=False
        )
    
    sessions = st.session_state.conference_db.get_sessions()
    presentations = st.session_state.conference_db.get_presentations()
    
    st.info(f"{len(sessions)} sessions to process")
    
    if st.button("Generate Titles", type="primary"):
        try:
            # Build kwargs based on backend
            kwargs = {}
            if backend == "gemini":
                actual_api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
                if actual_api_key:
                    kwargs["api_key"] = actual_api_key
            elif backend == "ollama":
                kwargs["host"] = ollama_host
            
            generator = create_title_generator(
                backend=backend,
                model=model,
                **kwargs
            )
            
            cached_generator = CachedTitleGenerator(
                generator, st.session_state.conference_db
            )
            
            progress_bar = st.progress(0)
            
            for i, session in enumerate(sessions):
                session_pres = [p for p in presentations 
                              if p.get("session_id") == session["session_id"]]
                
                if session_pres:
                    pres_data = [
                        {"title": p["title"], "abstract": p.get("abstract", ""),
                         "abstract_id": p["abstract_id"]}
                        for p in session_pres
                    ]
                    
                    result = cached_generator.generate(
                        session_id=session["session_id"],
                        presentations=pres_data,
                        force_regenerate=force_regenerate,
                    )
                    
                    st.write(f"**{session['session_id']}**: {result.titles[0] if result.titles else 'No title'}")
                
                progress_bar.progress((i + 1) / len(sessions))
            
            st.success("Titles generated!")
            
            config["default_title_model"] = model
            save_config(config)
            
        except Exception as e:
            st.error(f"Error generating titles: {e}")
    
    # Navigation
    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.current_step = 4
            st.rerun()
    with col2:
        if st.button("Next: Review & Export →", type="primary"):
            st.session_state.current_step = 6
            st.rerun()


def step_review_export():
    """Step 7: Review results and export."""
    st.header("📤 Review & Export")
    
    if st.session_state.conference_db is None:
        st.warning("No conference loaded")
        return
    
    # Show summary
    stats = st.session_state.conference_db.get_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Presentations", stats["total_presentations"])
    with col2:
        st.metric("Total Sessions", stats["total_sessions"])
    with col3:
        st.metric("Unplaced", stats["unplaced_presentations"])
    
    # Calculate overall metrics if we have embeddings
    if st.session_state.embeddings is not None and st.session_state.abstract_ids is not None:
        sessions = st.session_state.conference_db.get_sessions()
        if sessions:
            # Build session assignments from database
            session_assignments = {}
            for session in sessions:
                session_detail = st.session_state.conference_db.get_session(session["session_id"])
                if session_detail and "presentations" in session_detail:
                    for pres in session_detail["presentations"]:
                        session_assignments[pres["abstract_id"]] = session["session_id"]
            
            if session_assignments:
                try:
                    metrics = calculate_all_metrics(
                        st.session_state.embeddings,
                        st.session_state.abstract_ids,
                        session_assignments,
                    )
                    
                    st.subheader("Session Quality Metrics")
                    
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    with mcol1:
                        st.metric(
                            "Mean Coherence", 
                            f"{metrics['summary']['mean_coherence']:.3f}",
                            help="Average within-session similarity (higher = more related presentations)"
                        )
                    with mcol2:
                        st.metric(
                            "Mean Fit", 
                            f"{metrics['summary']['mean_fit']:.3f}",
                            help="Average presentation-to-session fit (higher = better individual placement)"
                        )
                    with mcol3:
                        st.metric(
                            "Min Coherence", 
                            f"{metrics['summary']['min_coherence']:.3f}",
                            help="Lowest session coherence (identifies weakest session)"
                        )
                    with mcol4:
                        st.metric(
                            "Max Coherence", 
                            f"{metrics['summary']['max_coherence']:.3f}",
                            help="Highest session coherence"
                        )
                except Exception as e:
                    st.warning(f"Could not calculate metrics: {e}")
    
    # Sessions table
    st.subheader("Sessions")
    sessions = st.session_state.conference_db.get_sessions()
    if sessions:
        df_sessions = pd.DataFrame(sessions)
        
        # Helper function to parse metric values that might be stored in various formats
        def parse_metric_value(x):
            """Parse metric value which might be float, string, list, or None."""
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "-"
            # If it's already a number, format it
            if isinstance(x, (int, float)):
                return f"{float(x):.3f}"
            # If it's a string, try to parse it
            if isinstance(x, str):
                x = x.strip()
                if x in ('-', '', 'None', 'null'):
                    return "-"
                try:
                    return f"{float(x):.3f}"
                except ValueError:
                    # Might be a comma-separated list or JSON
                    if ',' in x:
                        # Take first value
                        try:
                            first_val = x.split(',')[0].strip()
                            return f"{float(first_val):.3f}"
                        except ValueError:
                            return "-"
                    return "-"
            # If it's a list, take the first element
            if isinstance(x, (list, tuple)):
                if len(x) > 0:
                    try:
                        return f"{float(x[0]):.3f}"
                    except (ValueError, TypeError):
                        return "-"
            return "-"
        
        # Format coherence and distinctiveness as single number if they exist
        if "coherence" in df_sessions.columns:
            df_sessions["coherence"] = df_sessions["coherence"].apply(parse_metric_value)
        
        if "distinctiveness" in df_sessions.columns:
            df_sessions["distinctiveness"] = df_sessions["distinctiveness"].apply(parse_metric_value)
        
        # Select and rename columns for display
        display_cols = []
        col_rename = {}
        for col, name in [("session_id", "Session ID"), ("title", "Title"), 
                          ("coherence", "Coherence"), ("distinctiveness", "Distinctiveness"),
                          ("presentation_count", "# Presentations")]:
            if col in df_sessions.columns:
                display_cols.append(col)
                col_rename[col] = name
        
        st.dataframe(
            df_sessions[display_cols].rename(columns=col_rename), 
            width='stretch',
            hide_index=True
        )
    
    # Export options
    st.subheader("Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_profile = st.selectbox(
            "Export Profile",
            options=["cloud_viewer", "organizer_full", "room_assignment", "presenter_list"],
            format_func=lambda x: {
                "cloud_viewer": "Cloud Viewer (public, no PII)",
                "organizer_full": "Organizer Full (all data)",
                "room_assignment": "Room Assignment (spreadsheet)",
                "presenter_list": "Presenter List (contacts)",
            }.get(x, x)
        )
        
        export_dir = st.text_input(
            "Export Directory",
            value=str(Path.cwd() / "exports")
        )
    
    with col2:
        version_tag = st.text_input(
            "Version Tag",
            value=datetime.now().strftime("%Y%m%d")
        )
    
    if st.button("Export", type="primary"):
        try:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            
            if export_profile in ["cloud_viewer", "organizer_full"]:
                if st.session_state.embeddings is not None:
                    profile = PROFILE_CLOUD_VIEWER if export_profile == "cloud_viewer" else PROFILE_ORGANIZER_FULL
                    result = export_for_viewer(
                        st.session_state.conference_db,
                        st.session_state.embeddings,
                        st.session_state.abstract_ids,
                        export_path / version_tag,
                        profile=profile,
                        version_tag=version_tag,
                    )
                    st.success(f"Exported to {export_path / version_tag}")
                    st.json(result)
                else:
                    st.error("Embeddings required for this export type")
            else:
                # Spreadsheet export
                profile = PROFILE_ROOM_ASSIGNMENT if export_profile == "room_assignment" else ExportProfile(
                    name="custom", description="", format="excel"
                )
                output_file = export_path / f"{version_tag}_{export_profile}.xlsx"
                result = export_spreadsheet(
                    st.session_state.conference_db,
                    output_file,
                    profile=profile,
                )
                st.success(f"Exported to {output_file}")
                st.json(result)
                
        except Exception as e:
            st.error(f"Export error: {e}")
    
    # Navigation
    st.divider()
    if st.button("← Back"):
        st.session_state.current_step = 5
        st.rerun()


# === Main App ===

def render_data_viewer_main():
    """Render data viewer in main content area with tabs."""
    st.header("📊 Current Data Viewer")
    
    if st.session_state.conference_db is None:
        st.warning("No conference loaded. Please set up a conference first.")
        return
    
    # Create tabs for different data views
    tab_pres, tab_hybrid, tab_committees, tab_sessions = st.tabs([
        "👥 Presentations", 
        "🔄 Hybrid Sessions (Input)", 
        "🏛️ Committees",
        "📋 Created Sessions"
    ])
    
    with tab_pres:
        st.subheader("Imported Presentations")
        presentations = st.session_state.conference_db.get_presentations()
        if presentations:
            df_pres = pd.DataFrame(presentations)
            st.info(f"{len(presentations)} presentations loaded")
            
            # Show all available columns
            st.dataframe(
                df_pres,
                hide_index=True,
                height=400,
                width='stretch',
            )
        else:
            st.info("No presentations imported yet. Use the Import Data step to load presentations.")
    
    with tab_hybrid:
        st.subheader("Hybrid Sessions (Input)")
        st.write("Hybrid sessions are pre-defined sessions (e.g., invited or special sessions) that presentations can be assigned to.")
        try:
            # Query sessions table for hybrid sessions (is_hybrid = 1)
            with sqlite3.connect(st.session_state.conference_db.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM sessions WHERE is_hybrid = 1"
                )
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                if rows:
                    df_hybrid = pd.DataFrame(rows, columns=columns)
                    st.info(f"{len(rows)} hybrid sessions")
                    st.dataframe(df_hybrid, hide_index=True, height=400, width='stretch')
                else:
                    st.info("No hybrid sessions found. Hybrid sessions can be imported or created with the is_hybrid flag.")
        except Exception as e:
            st.error(f"Error loading hybrid sessions: {e}")
    
    with tab_committees:
        st.subheader("Committees")
        try:
            committees = st.session_state.conference_db.get_committees()
            if committees:
                df_committees = pd.DataFrame(committees)
                st.info(f"{len(committees)} committees")
                st.dataframe(df_committees, hide_index=True, height=400, width='stretch')
            else:
                st.info("No committees imported yet. Use the Import Data step to load committees.")
        except Exception as e:
            st.error(f"Error loading committees: {e}")
    
    with tab_sessions:
        st.subheader("Created Sessions")
        sessions = st.session_state.conference_db.get_sessions()
        if sessions:
            df_sessions = pd.DataFrame(sessions)
            st.info(f"{len(sessions)} sessions created")
            st.dataframe(
                df_sessions,
                hide_index=True,
                height=400,
                width='stretch',
            )
            
            # Show session details with presentations
            st.divider()
            st.subheader("Session Details")
            session_ids = [s["session_id"] for s in sessions]
            selected_session = st.selectbox("Select a session to view presentations:", session_ids)
            
            if selected_session:
                session_detail = st.session_state.conference_db.get_session(selected_session)
                if session_detail and "presentations" in session_detail:
                    pres_list = session_detail["presentations"]
                    if pres_list:
                        st.write(f"**{len(pres_list)} presentations in this session:**")
                        df_session_pres = pd.DataFrame(pres_list)
                        st.dataframe(df_session_pres, hide_index=True, width='stretch')
                    else:
                        st.info("No presentations assigned to this session.")
        else:
            st.info("No sessions created yet. Use the Create Sessions step to generate sessions.")


def main():
    """Main application entry point."""
    init_session_state()
    
    # Initialize viewing_data state if not present
    if "viewing_data" not in st.session_state:
        st.session_state.viewing_data = False
    
    # Sidebar navigation
    st.sidebar.title("🎯 SMART")
    st.sidebar.caption("Session Matching And Automated Recommendation Tool")
    
    # View Current Data button at the top
    if st.sidebar.button("📊 View Current Data", use_container_width=True):
        st.session_state.viewing_data = True
        st.rerun()
    
    # Show database stats if available (compact, under the button)
    if st.session_state.conference_db:
        stats = st.session_state.conference_db.get_stats()
        st.sidebar.caption(f"{stats['total_presentations']} presentations | {stats['total_sessions']} sessions")
    
    st.sidebar.divider()
    
    steps = [
        "📁 Conference Setup",
        "📥 Import Data",
        "🧠 Generate Embeddings",
        "🔍 Check Duplicates",
        "📊 Create Sessions",
        "✨ Generate Titles",
        "📤 Review & Export",
    ]
    
    # Show step status
    for i, step in enumerate(steps):
        if i < st.session_state.current_step:
            st.sidebar.success(step)
        elif i == st.session_state.current_step:
            st.sidebar.info(f"**{step}** ←")
        else:
            st.sidebar.text(step)
    
    # Step selection (for jumping)
    st.sidebar.divider()
    selected_step = st.sidebar.selectbox(
        "Jump to step",
        options=list(range(len(steps))),
        format_func=lambda x: steps[x],
        index=st.session_state.current_step,
    )
    if selected_step != st.session_state.current_step:
        st.session_state.current_step = selected_step
        st.session_state.viewing_data = False
        st.rerun()
    
    # Render current view
    if st.session_state.viewing_data:
        # Show data viewer with back button
        if st.button("← Back to Processing Steps"):
            st.session_state.viewing_data = False
            st.rerun()
        render_data_viewer_main()
    else:
        # Render current step
        step_functions = [
            step_conference_setup,
            step_import_data,
            step_embeddings,
            step_check_duplicates,
            step_create_sessions,
            step_generate_titles,
            step_review_export,
        ]
        
        step_functions[st.session_state.current_step]()


if __name__ == "__main__":
    main()

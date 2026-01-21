"""
Example: Using the Refactored Embedding Workflow

This example demonstrates the new embedding system that stores embeddings
directly in the presentations dataframe and uses parquet format for persistence.
"""

import pandas as pd
from session_organizer import (
    load_presentations,
    embed_with_resume,
    remove_duplicates,
    extract_embeddings_dataframe,
    create_sessions_w_hybrid
)

# ============================================================================
# STEP 1: Load presentations
# ============================================================================
df_presentations, title_col, abstract_col, id_col, topic_col = load_presentations(
    file_path='Submissions_852925_all.csv',
    Title_name='Title',
    Abstract_name='Abstract',
    Abstract_ID_name='Submission ID',
    title_column='Title',
    abstract_column='Abstract',
    abstract_id_column='Abstract ID',
    topic_column='Title and Abstract'
)

print(f"Loaded {len(df_presentations)} presentations")
print(f"Columns: {df_presentations.columns.tolist()}")

# ============================================================================
# STEP 2: Embed presentations with resume capability
# ============================================================================
# The first time: will embed all presentations and save to parquet
# Subsequent times: will load cached embeddings and only embed missing ones
# If presentation text changes, it will detect and re-embed
df_presentations = embed_with_resume(
    df_presentations=df_presentations,
    topic_column=topic_col,
    model_name='embedding-001',
    embedding_model_name='gemini-embedding-001',
    backup_filename='presentations_with_embeddings.parquet',
    delay_seconds=0.1  # Small delay between API calls to avoid rate limits
)

print(f"\nEmbeddings status:")
print(f"  Total presentations: {len(df_presentations)}")
print(f"  Embedded: {df_presentations['embedding'].notna().sum()}")
print(f"  Missing: {df_presentations['embedding'].isna().sum()}")

# ============================================================================
# STEP 3: Remove near-duplicates (optional)
# ============================================================================
# The function now works directly with the embedding column in df_presentations
# It returns both the cleaned presentations and a legacy embeddings dataframe
# for backwards compatibility with clustering functions

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
similarity_func = lambda a, b: util.pytorch_cos_sim(a, b)

df_presentations, df_embeddings = remove_duplicates(
    df_presentations=df_presentations,
    similarity_func=similarity_func,
    embedding_column='embedding',
    embedding_model_name='gemini-embedding-001',
    threshold=0.95
)

print(f"\nAfter removing duplicates: {len(df_presentations)} presentations")

# ============================================================================
# STEP 4: Extract embeddings dataframe for clustering
# ============================================================================
# For downstream functions like create_sessions_w_hybrid() that expect
# the legacy embeddings dataframe format, use extract_embeddings_dataframe()
# Note: This is already done automatically by remove_duplicates()

# If you want to extract embeddings manually:
df_embeddings = extract_embeddings_dataframe(
    df_presentations=df_presentations,
    embedding_model_name='gemini-embedding-001',
    embedding_column='embedding'
)

print(f"\nEmbeddings dataframe shape: {df_embeddings.shape}")
print(f"Columns: {df_embeddings.columns.tolist()}")

# ============================================================================
# STEP 5: Create sessions using clustering
# ============================================================================
df_sessions, labels, metadata = create_sessions_w_hybrid(
    df_presentations=df_presentations,
    similarity_func=similarity_func,
    df_presentation_embeddings=df_embeddings,
    max_sessions=50,
    min_session_size=8,
    tree_merge_stop=0.95
)

print(f"\nClustering results:")
print(f"  Sessions created: {metadata['n_clusters']}")
print(f"  Presentations assigned: {metadata['n_assigned_items']}")
print(f"  Presentations unassigned: {metadata['n_unassigned_items']}")

# ============================================================================
# Key Benefits of New Workflow
# ============================================================================
# 1. No text-embedding mismatches: embeddings stored with presentations
# 2. Automatic resume: detects unchanged text and skips re-embedding
# 3. Change detection: changed text triggers re-embedding
# 4. Single file storage: parquet file contains everything
# 5. Backwards compatible: works with existing clustering functions

# ============================================================================
# Storage Details
# ============================================================================
# The parquet file (presentations_with_embeddings.parquet) contains:
#   - All original presentation columns (Title, Abstract, etc.)
#   - 'embedding' column: numpy array (768-dim for Gemini model)
#   - All other data columns
# 
# File size: ~10-20MB for 1000 presentations with embeddings
# Compression: snappy (faster than gzip, smaller than uncompressed)

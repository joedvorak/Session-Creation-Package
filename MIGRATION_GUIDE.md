# Migration Guide: Embedding Workflow Refactor

## Quick Reference

### Before (Old Pattern)
```python
# Separate embeddings dataframe - risk of mismatch
df_embeddings = embed_documents_with_genai(df_presentations, topic_col, model, df_embeddings=None)
save_embeddings_to_file(df_embeddings, "embeddings_backup.pkl")
loaded_embeddings = load_embeddings_from_file("embeddings_backup.pkl")
```

### After (New Pattern)
```python
# Embeddings integrated into presentations - no mismatch risk
df_presentations = embed_with_resume(df_presentations, topic_col, model)
# Automatically saved to presentations_with_embeddings.parquet
```

## Function Signature Changes

### `embed_with_resume()`
**Old:**
```python
embed_with_resume(df_presentations, topic_column, model_name, 
                  backup_filename="embeddings_backup.pkl", delay_seconds=0)
```

**New:**
```python
embed_with_resume(df_presentations, topic_column, model_name, 
                  embedding_model_name=None,
                  backup_filename="presentations_with_embeddings.parquet", 
                  delay_seconds=0)
```

**Changes:**
- Added `embedding_model_name` parameter (optional, defaults to model_name)
- Default backup filename changed from pickle to parquet
- Return value: `df_presentations` with embedding column (not separate dataframe)

### `remove_duplicates()`
**Old:**
```python
remove_duplicates(df_presentations, df_embeddings, similarity_func, threshold=0.95)
# Returns: (df_presentations, df_embeddings)
```

**New:**
```python
remove_duplicates(df_presentations, similarity_func, 
                  embedding_column='embedding', embedding_model_name=None, threshold=0.95)
# Returns: (df_presentations, df_embeddings_legacy)
```

**Changes:**
- Second parameter changed from `df_embeddings` to `similarity_func`
- Added `embedding_column` parameter (default='embedding')
- Added `embedding_model_name` parameter for tracking
- Automatically extracts legacy embeddings format (backwards compatible)

### New Functions
```python
# For backwards compatibility with clustering functions
extract_embeddings_dataframe(df_presentations, embedding_model_name, embedding_column='embedding')

# For parquet storage (replaces pickle functions)
save_presentations_with_embeddings(df_presentations, filename="presentations_with_embeddings.parquet")
load_presentations_with_embeddings(filename="presentations_with_embeddings.parquet")
```

## Step-by-Step Migration

### If you have existing code using old pattern:

**Step 1: Replace embedding loading/saving**
```python
# OLD
df_embeddings = embed_documents_with_genai(df_presentations, topic_col, model, df_embeddings=None)
save_embeddings_to_file(df_embeddings, "old_embeddings.pkl")
loaded = load_embeddings_from_file("old_embeddings.pkl")

# NEW
df_presentations = embed_with_resume(df_presentations, topic_col, model)
# That's it! Automatically saved to parquet, resume is built-in
```

**Step 2: Update remove_duplicates calls**
```python
# OLD
df_presentations, df_embeddings = remove_duplicates(df_presentations, df_embeddings, similarity_func)

# NEW
df_presentations, df_embeddings = remove_duplicates(df_presentations, similarity_func, 
                                                     embedding_column='embedding')
```

**Step 3: For clustering, extract legacy format if needed**
```python
# If your clustering code expects the old embeddings dataframe format:
df_embeddings = extract_embeddings_dataframe(df_presentations, "gemini-embedding-001")

# Then use as before with create_sessions_w_hybrid()
df_sessions, labels, metadata = create_sessions_w_hybrid(
    df_presentations, similarity_func, df_embeddings, ...
)
```

## Data Structure Changes

### Embeddings Storage

**Old approach:**
- Separate file: `embeddings_backup.pkl`
- Structure: 
  ```
  topic  | embedding | status
  ...    | [vec]     | 'success'
  ```
- Risk: Text and embedding could get out of sync

**New approach:**
- Single file: `presentations_with_embeddings.parquet`
- Structure:
  ```
  Title | Abstract | ... | embedding | (other columns)
  ...   | ...      | ... | [vec]     | ...
  ```
- Benefit: Text and embedding always together

### Column Information

**df_presentations after `embed_with_resume()`:**
- Contains new `embedding` column with numpy arrays
- Size per embedding: ~768 dimensions for Gemini model
- Type: object (contains arrays)
- Null check: `df_presentations['embedding'].isna()` to find missing embeddings

**df_embeddings after `extract_embeddings_dataframe()`:**
- `dim_0`, `dim_1`, ..., `dim_767`: Individual embedding dimensions
- `embedding_model`: Model name string
- Index: Matches df_presentations index
- Format: Ready for scikit-learn and similarity functions

## Testing Your Migration

```python
# Test 1: Verify embeddings are stored in presentations
assert 'embedding' in df_presentations.columns
assert df_presentations['embedding'].notna().all()

# Test 2: Verify parquet file is created
import os
assert os.path.exists('presentations_with_embeddings.parquet')

# Test 3: Verify resume works (should be fast on second run)
df = embed_with_resume(df_presentations, topic_col, model)
# Should show "Already embedded: N presentations"

# Test 4: Verify change detection
df_copy = df.copy()
df_copy.at[0, topic_col] = 'MODIFIED TEXT'
df_modified = embed_with_resume(df_copy, topic_col, model)
# Should detect new text and re-embed first presentation

# Test 5: Verify backwards compatibility
df_emb = extract_embeddings_dataframe(df_presentations, "gemini-embedding-001")
assert COLUMNS['EMBEDDING_MODEL'] in df_emb.columns
assert df_emb.shape[1] > 768  # 768 dims + model column
```

## Troubleshooting

### Issue: "Column 'embedding' not found"
**Solution:** Make sure you're using the new `embed_with_resume()` function which adds the column.

### Issue: "Cannot extract embeddings: N presentations have null embeddings"
**Solution:** Make sure all embeddings are complete before calling `extract_embeddings_dataframe()`. Use `embed_with_resume()` to fill missing ones.

### Issue: Old pickle files won't load
**Solution:** This is intentional per Option A (auto-regenerate). Re-run `embed_with_resume()` which will recreate the parquet file.

### Issue: Resume isn't working (re-embedding everything)
**Solution:** Check that:
1. Parquet file exists: `presentations_with_embeddings.parquet`
2. Topic column name matches exactly
3. Topic text is identical (exact match, no fuzzy matching)

## Performance Notes

- **First run:** Same speed as before (all presentations embedded)
- **Resume run (no changes):** ~1s (just loads parquet)
- **Resume run (partial changes):** Proportional to number of new presentations
- **File size:** ~10-20MB for 1000 presentations (snappy compression)
- **RAM:** One presentation cached at a time (streaming-friendly)

## Long-term Roadmap

Current implementation:
- ✅ Embeddings in presentations dataframe
- ✅ Parquet storage
- ✅ Exact-match resume
- ✅ Backwards compatible via `extract_embeddings_dataframe()`

Future directions:
- [ ] Remove separate embeddings dataframe entirely
- [ ] Update clustering functions to work with embedding column directly
- [ ] Add embedding metadata (timestamp, API version, etc.)
- [ ] Support model version migrations

import pandas as pd
from sentence_transformers import SentenceTransformer
import random
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pickle
from scipy.cluster.hierarchy import linkage
from google import genai
from google.genai import types
from pydantic import BaseModel
import time
import os
from dotenv import load_dotenv
from itertools import chain
from sklearn.metrics import silhouette_samples

COLUMNS = {
    'EMBEDDING_MODEL': 'embedding_model',
    'SESSION_SIZE': 'session_size',
    'GEN_PRESENTATION_INDICES': 'gen_presentation_indices',
    'CLUSTER_ID': 'cluster_id',
    'HYBRID_SESSION_TITLE': 'hybrid_session_title',
    'HYBRID_INVITED_PRESENTATIONS': 'hybrid_invited_presentations',
    'SESSION_TITLE': 'session_title',
    'FINAL_SESSION_TITLE': 'final_session_title',
}

UNSET_SESSION_TITLE_TEXT = "Not Set Yet"

#Helper functions
def create_index_mappings(df):
    """
    Create bidirectional mappings between DataFrame indices and array positions.
    
    Args:
        df: DataFrame to create mappings for
        
    Returns:
        tuple: (position_to_index, index_to_position)
    """
    position_to_index = {pos: idx for pos, idx in enumerate(df.index)}
    index_to_position = {idx: pos for pos, idx in enumerate(df.index)}
    return position_to_index, index_to_position

def array_positions_to_df_indices(array_positions, position_to_index_map):
    """
    Convert array positions to DataFrame indices.
    
    Args:
        array_positions: List/array of positions in the array
        position_to_index_map: Mapping from positions to DataFrame indices
        
    Returns:
        list: Corresponding DataFrame indices
    """
    if isinstance(array_positions, (int, np.integer)):
        return position_to_index_map[array_positions]
    
    return [position_to_index_map[pos] for pos in array_positions]

def df_indices_to_array_positions(df_indices, index_to_position_map):
    """
    Convert DataFrame indices to array positions.
    
    Args:
        df_indices: List/array of DataFrame indices  
        index_to_position_map: Mapping from DataFrame indices to positions
        
    Returns:
        list: Corresponding array positions
    """
    if isinstance(df_indices, (int, np.integer)):
        return index_to_position_map[df_indices]
    
    return [index_to_position_map[idx] for idx in df_indices]


def load_presentations(file_path, Title_name='Title', Abstract_name='Abstract', Abstract_ID_name='Submission ID', 
                       title_column='Title', abstract_column='Abstract', abstract_id_column='Abstract ID', topic_column='Title and Abstract'):
    """
    Load presentations from an Excel file.
    Args:
        file_path (str): Path to the Excel file.
        Title_name (str): Spreadsheet column name that contains the titles.
        Abstract_name (str): Spreadsheet column name that contains the abstracts.
        Abstract_ID_name (str): Spreadsheet column name that contains the abstract IDs.
        title_column (str): Name for the title column in the output DataFrame.
        abstract_column (str): Name for the abstract column in the output DataFrame.
        abstract_id_column (str): Name for the abstract ID column in the output DataFrame.
        topic_column (str): Name for the combined title and abstract column in the output DataFrame.
    Returns:
        tuple: (df, title_column, abstract_column, abstract_id_column, topic_column)
    """
    # Determine file type and read accordingly
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format. Please use CSV (.csv) or Excel (.xlsx, .xls) files.")
    # Validate required columns exist in the file
    if Title_name not in df.columns or Abstract_name not in df.columns or Abstract_ID_name not in df.columns:
        raise ValueError(f"Columns '{Title_name}', '{Abstract_name}', '{Abstract_ID_name}' must be present in the Excel file.")
    
    # Create column rename map for the key columns
    column_rename_map = {
        Title_name: title_column,
        Abstract_name: abstract_column,
        Abstract_ID_name: abstract_id_column
    }
    
    # Rename the key columns while keeping all other columns
    df = df.rename(columns=column_rename_map)

    # Drop any presentations that are missing an abstract or title
    df = df.dropna(subset=[title_column, abstract_column])
    
    # Combine Titles and Abstracts with a colon in between. 
    # This should be the same as + but .agg() handles empty fields or fields that have non-text entries.
    df[topic_column] = df[[title_column, abstract_column]].agg(': '.join, axis=1)
    
    return df, title_column, abstract_column, abstract_id_column, topic_column

def load_committees(file_path, Committee_Name_column='Committee_Name', Description_column='Description',
                   committee_name_column='Committee_Name', description_column='Description', 
                   combined_column='Name_Description'):
    """
    Load committees from a CSV or Excel spreadsheet.
    
    Args:
        file_path (str): Path to the CSV or Excel file containing committee information
        Committee_Name_column (str): Spreadsheet column name that contains the committee names
        Description_column (str): Spreadsheet column name that contains the descriptions
        committee_name_column (str): Name for the committee name column in the output DataFrame
        description_column (str): Name for the description column in the output DataFrame
        combined_column (str): Name for the combined committee name and description column
        
    Returns:
        tuple: (df_committees, committee_name_column, description_column, combined_column)
    """
    # Determine file type and read accordingly
    if file_path.lower().endswith('.csv'):
        df_committees = pd.read_csv(file_path)
    elif file_path.lower().endswith(('.xlsx', '.xls')):
        df_committees = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format. Please use CSV (.csv) or Excel (.xlsx, .xls) files.")
    
    # Validate required columns exist in the file
    if Committee_Name_column not in df_committees.columns:
        raise ValueError(f"Column '{Committee_Name_column}' not found in file. Available columns: {list(df_committees.columns)}")
    if Description_column not in df_committees.columns:
        raise ValueError(f"Column '{Description_column}' not found in file. Available columns: {list(df_committees.columns)}")
    
    # Select and rename columns
    df_committees = df_committees[[Committee_Name_column, Description_column]].rename(
        columns={
            Committee_Name_column: committee_name_column, 
            Description_column: description_column
        }
    )
    
    # Clean up any NaN values
    df_committees = df_committees.dropna(subset=[committee_name_column, description_column])
    
    # Create combined description column (like topic_column in load_presentations)
    df_committees[combined_column] = df_committees[[committee_name_column, description_column]].agg(': '.join, axis=1)
    
    return df_committees, committee_name_column, description_column, combined_column

def embed_documents(df_presentations, topic_column, embedding_model):
    if not isinstance(embedding_model, SentenceTransformer):
        raise ValueError("embedding_model must be an instance of SentenceTransformer.")
    if topic_column not in df_presentations.columns:
        raise ValueError(f"topic_column '{topic_column}' must be present in the DataFrame.")
    
    # Extract model name for the new column
    model_name = getattr(embedding_model, 'model_name_or_path', 'Unknown')
    if hasattr(embedding_model, 'model_card_data') and embedding_model.model_card_data:
        base_model = getattr(embedding_model.model_card_data, 'base_model', None)
        if base_model:
            model_info = f"{model_name} ({base_model})"
        else:
            model_info = model_name
    else:
        model_info = model_name
    
    if getattr(embedding_model.model_card_data, "base_model", None) in ["jxm/cde-small-v1", "jxm/cde-small-v2"]:
        # This is the CDE model, so we need to use the special CDE embedding function
        return cde_embed_documents(df_presentations, topic_column, embedding_model, model_info)
    else:
        # This is a standard SentenceTransformer model, so we can use the standard embedding function
        return standard_embed_documents(df_presentations, topic_column, embedding_model, model_info)

def standard_embed_documents(df_presentations, topic_column, embedding_model, model_info):
    """
    Embed the presentation topics using a standard SentenceTransformer model.
    
    Args:
        df_presentations (pd.DataFrame): DataFrame containing the presentations.
        topic_column (str): Column name in df_presentations that contains the topics to embed.
        embedding_model (SentenceTransformer): The SentenceTransformer model to use for embedding.
        model_info (str): Information about the embedding model used.
        
    Returns:
        pd.DataFrame: DataFrame containing the embeddings of the presentation topics with model info.
    """
    # Text to embed
    presentation_topics = df_presentations[topic_column].tolist()
    # Embed the titles and abstracts using the SentenceTransformer model
    presentation_embeddings = embedding_model.encode(
        presentation_topics,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    df_presentation_embeddings = pd.DataFrame(presentation_embeddings.cpu(), index = df_presentations.index)
    
    # Add model information column
    df_presentation_embeddings[COLUMNS['EMBEDDING_MODEL']] = model_info
    
    return df_presentation_embeddings

def cde_embed_documents(df_presentations, topic_column, cde_embeddings_model, model_info):
    """
    Embed the presentation topics using a CDE (Contextual Document Embeddings) model.
    Args:
        df_presentations (pd.DataFrame): DataFrame containing the presentations.
        topic_column (str): Column name in df_presentations that contains the topics to embed.
        cde_embeddings_model (SentenceTransformer): The CDE SentenceTransformer model to use for embedding.
        model_info (str): Information about the embedding model used.
    Returns:
        pd.DataFrame: DataFrame containing the embeddings of the presentation topics with model info.
    """
    # Text to embed
    presentation_topics = df_presentations[topic_column].tolist()

    # Embed the titles and abstracts
    # First, create a minicorpus of as required by the CDE model.
    minicorpus_size = cde_embeddings_model[0].config.transductive_corpus_size
    # Get the unique elements from the original population
    #    Using list(set(...)) ensures uniqueness and handles potential duplicates in presentation_topics
    unique_docs = list(set(presentation_topics))
    num_unique = len(unique_docs)
    if minicorpus_size < num_unique:
        # If the minicorpus size is smaller than the number of unique documents, sample without replacement
        minicorpus_docs = random.sample(presentation_topics, k=minicorpus_size)
    else: 
        # Start the minicorpus with all unique documents
        minicorpus_docs = list(unique_docs) # Make a copy to start

        # Calculate how many more documents are needed
        remaining_needed = minicorpus_size - num_unique

        # If more are needed, sample *with replacement* from the *original* population
        if remaining_needed > 0:
            # Sample the remaining items randomly WITH replacement from the original presentation_topics
            additional_docs = random.choices(presentation_topics, k=remaining_needed)
            # Add these to the minicorpus
            minicorpus_docs.extend(additional_docs)
    assert len(minicorpus_docs) == minicorpus_size # You must use exactly this many documents in the minicorpus. You can oversample if your corpus is smaller.

    cde_dataset_embeddings = cde_embeddings_model.encode(
        minicorpus_docs,
        prompt_name="document",
        convert_to_tensor=True,
        show_progress_bar=True
    )

    # Now embed the titles and abstracts using the CDE embeddings model
    presentation_embeddings = cde_embeddings_model.encode(
        presentation_topics,
        prompt_name="document",
        dataset_embeddings=cde_dataset_embeddings,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    df_presentation_embeddings = pd.DataFrame(presentation_embeddings.cpu(), index = df_presentations.index)

    # Add model information column
    df_presentation_embeddings[COLUMNS['EMBEDDING_MODEL']] = model_info

    return df_presentation_embeddings

def remove_duplicates(df_presentations, df_embeddings, similarity_func, threshold=0.95):
    """
    Remove near-duplicate rows based on a similarity threshold.
    Args:
        df_presentations (pd.DataFrame): DataFrame containing presentation data.
        df_embeddings (pd.DataFrame): DataFrame containing presentation embeddings.
        similarity_func (callable): Function to compute similarity between embeddings.
        threshold (float): Similarity threshold for considering items as near duplicates.
    Returns:
        pd.DataFrame: Presentations DataFrame with near-duplicate rows removed.
        pd.DataFrame: Embeddings DataFrame with near-duplicate rows removed.
    """
    # Get the indices from the dataframe (important for referencing)
    presentation_indices = df_presentations.index.tolist()
    embeddings_array = df_embeddings.drop(columns=[COLUMNS['EMBEDDING_MODEL']]).values
    
    # Create a set to store the indices we want to REMOVE
    indices_to_remove = set()
    
    # Calculate the similarity matrix for the embeddings
    similarity_matrix_np = similarity_func(embeddings_array, embeddings_array)
    
    # Convert to numpy if it's a torch tensor
    if hasattr(similarity_matrix_np, 'cpu'):
        similarity_matrix_np = similarity_matrix_np.cpu().numpy()
    elif hasattr(similarity_matrix_np, 'numpy'):
        similarity_matrix_np = similarity_matrix_np.numpy()
    
    # Iterate through the upper triangle of the similarity matrix
    num_items = similarity_matrix_np.shape[0]

    for i in range(num_items):
        for j in range(i + 1, num_items):
            if similarity_matrix_np[i, j] >= threshold:
                # Get the actual DataFrame indices for positions i and j
                idx_i = presentation_indices[i]
                idx_j = presentation_indices[j]
                print(f"Near duplicate found: Index {idx_i} and Index {idx_j} (Similarity: {similarity_matrix_np[i, j]:.4f}).")
                # Remove the item with the LOWER index (keep the higher one)
                if idx_i < idx_j:
                    indices_to_remove.add(idx_i)
                else:
                    indices_to_remove.add(idx_j)

    # Convert the set of indices to remove into a list
    indices_to_remove_list = sorted(list(indices_to_remove))

    print(f"\nFound {len(indices_to_remove_list)} near-duplicate presentations to remove (keeping highest index).")
    print(f"Indices to remove: {indices_to_remove_list}")

    # --- Perform the removal ---
    df_presentations = df_presentations.drop(index=indices_to_remove_list)
    df_embeddings = df_embeddings.drop(index=indices_to_remove_list)

    # --- Verification ---
    print(f"\nFinal number of oral presentations: {len(df_presentations)}")
    print(f"Final shape of embeddings matrix: {df_embeddings.shape}")

    # Verify indices still match
    if not df_presentations.index.equals(df_embeddings.index):
        raise ValueError("DataFrame indices don't match after duplicate removal!")

    # Reset the index of the DataFrames to ensure they are clean and sequential
    df_presentations = df_presentations.reset_index(drop=True)
    df_embeddings = df_embeddings.reset_index(drop=True)
    
    return df_presentations, df_embeddings

def get_unique_top_indices_variable(data_array, target_counts):
    """
    Efficiently selects variable numbers of top unique indices for each row of an array.

    This function implements a priority-based two-phase process with support for
    different target counts per row:
    1. Priority Assignment: Uses a priority matrix to determine which row has 
       the highest claim on each column. For each row, assigns columns from 
       its top candidates that it has the highest priority for.
    2. Greedy Backfilling: For rows that still need more indices, greedily 
       assigns the best remaining unassigned columns in value-descending order.

    The algorithm uses tie-breaking to ensure deterministic results when multiple
    rows have identical values for the same column (lower row indices win).

    Args:
        data_array (np.ndarray): The 2D input array of values.
        target_counts (array-like): Array/list of desired number of unique indices 
                                   per row. Must have same length as number of rows.

    Returns:
        list: A list where each element is a 1D numpy array containing the 
              unique column indices for that row, sorted in ascending order.
              Each array's length equals the corresponding target_counts value.

    Algorithm Details:
        - Uses argpartition for O(n) top-k selection per row
        - Maintains assignment tracking with boolean arrays for efficiency  
        - Processes remaining columns in batches to minimize redundant operations
        - Time complexity: O(nm log m) where n=rows, m=columns
        
    Raises:
        ValueError: If target_counts length doesn't match number of rows or
                   if any target count exceeds available columns.
    """
    target_counts = np.asarray(target_counts)
    num_rows, num_cols = data_array.shape
    
    # Validation
    if len(target_counts) != num_rows:
        raise ValueError(f"target_counts length ({len(target_counts)}) must match number of rows ({num_rows})")
    
    if np.any(target_counts > num_cols):
        raise ValueError(f"target_counts cannot exceed number of columns ({num_cols})")
    
    if np.any(target_counts < 0):
        raise ValueError("target_counts must be non-negative")
    
    # Handle edge case where some rows need 0 indices
    max_target = np.max(target_counts) if len(target_counts) > 0 else 0
    if max_target == 0:
        return [np.array([], dtype=int) for _ in range(num_rows)]
    
    # Create priority matrix: higher values = higher priority
    # Break ties by giving priority to lower row indices
    priority_matrix = data_array + (np.arange(num_rows)[:, None] * 1e-10)
    
    # Find winning row for each column
    winning_rows = np.argmax(priority_matrix, axis=0)
    
    # Create initial assignments based on top values per row
    # Use the maximum target count for argpartition to ensure we get enough candidates
    top_indices = {}
    for row in range(num_rows):
        if target_counts[row] > 0:
            k = min(target_counts[row], num_cols)
            top_indices[row] = np.argpartition(data_array[row], -k)[-k:]
        else:
            top_indices[row] = np.array([], dtype=int)
    
    # Build result efficiently - use list since rows have different lengths
    result = [np.full(target_counts[row], -1, dtype=int) for row in range(num_rows)]
    assigned_cols = np.zeros(num_cols, dtype=bool)
    row_counts = np.zeros(num_rows, dtype=int)
    
    # Phase 1: Assign columns to their winning rows if they're in top candidates
    for row in range(num_rows):
        if target_counts[row] == 0:
            continue
            
        row_top_indices = top_indices[row]
        if len(row_top_indices) == 0:
            continue
            
        row_values = data_array[row, row_top_indices]
        
        # Sort by value (descending)
        sorted_order = np.argsort(row_values)[::-1]
        
        for idx in sorted_order:
            col = row_top_indices[idx]
            if (not assigned_cols[col] and 
                winning_rows[col] == row and 
                row_counts[row] < target_counts[row]):
                result[row][row_counts[row]] = col
                assigned_cols[col] = True
                row_counts[row] += 1
    
    # Phase 2: Fill remaining slots with best available columns
    remaining_cols = np.where(~assigned_cols)[0]
    
    for row in range(num_rows):
        need = target_counts[row] - row_counts[row]
        if need > 0 and len(remaining_cols) > 0:
            # Get values for remaining columns for this row
            remaining_values = data_array[row, remaining_cols]
            
            # Get top 'need' columns
            if len(remaining_cols) >= need:
                best_remaining = np.argpartition(remaining_values, -need)[-need:]
            else:
                best_remaining = np.arange(len(remaining_cols))
            
            # Sort by value (descending)
            best_remaining = best_remaining[np.argsort(remaining_values[best_remaining])[::-1]]
            
            # Assign them
            for i, idx in enumerate(best_remaining):
                if row_counts[row] >= target_counts[row]:
                    break
                col = remaining_cols[idx]
                result[row][row_counts[row]] = col
                row_counts[row] += 1
            
            # Remove assigned columns from remaining pool
            mask = np.ones(len(remaining_cols), dtype=bool)
            mask[best_remaining[:len(best_remaining)]] = False
            remaining_cols = remaining_cols[mask]
    
    # Sort each row's indices and convert to proper arrays
    for row in range(num_rows):
        if target_counts[row] > 0:
            # Only sort non-negative values (in case some slots weren't filled)
            valid_indices = result[row][result[row] >= 0]
            result[row] = np.sort(valid_indices)
        else:
            result[row] = np.array([], dtype=int)
    
    return result

def create_sessions_w_hybrid(df_presentations, similarity_func, df_presentation_embeddings, 
                             df_hybrid_presentations=None, hybrid_session_column=None, df_hybrid_embeddings=None, 
                   max_sessions=100, min_session_size=8, tree_merge_stop=0.95, cluster_column_name="Session Code",
                   final_session_title_column=COLUMNS['FINAL_SESSION_TITLE']):
    """
    Create sessions from the presentations based on their embeddings and similarities. It is optional to consider hybrid sessions.
    If hybrid sessions are provided, they will be filled first before clustering the remaining presentations. This means that hybrid
    sessions always get first pick on matching presentations.

    Args:
        df_presentations (pd.DataFrame): DataFrame containing the presentations.
        similarity_func (callable): Function from embedding_model.similarity to compute the similarity matrix of the presentation embeddings.
        df_hybrid_presentations (pd.DataFrame, optional): DataFrame containing hybrid presentations. If provided, these will be filled first.
        hybrid_session_column (str, optional): Column name in df_hybrid_presentations that contains the session information.
        df_hybrid_embeddings (pd.DataFrame, optional): DataFrame containing the embeddings of the hybrid presentations. If provided, these will be used for hybrid sessions.
                                                    The index of df_hybrid_embeddings must align with df_hybrid_presentations.
                                                    The last column is the embedding model name.
        df_presentation_embeddings (pd.DataFrame): DataFrame containing the embeddings of the presentation topics.
                                                 The last column is the embedding model name.
        max_sessions (int): Maximum number of sessions to create.
        min_session_size (int): Minimum number of presentations in a session.
        tree_merge_stop (float): The fraction of the tree to stop clustering at.
        cluster_column_name (str): The name of the column to store cluster labels.
        final_session_title_column (str): The name of the column to store the final session titles.

    Returns:
        tuple: (df_presentations, df_sessions, labels, metadata)
    Raises:
        ValueError: If parameters are invalid.
    """
    # Input validation
    if max_sessions <= 0:
        raise ValueError("max_sessions must be greater than 0.")
    if not (0 < tree_merge_stop <= 1):
        raise ValueError("tree_merge_stop must be between 0 and 1.")
    if min_session_size < 1:
        raise ValueError("min_session_size must be at least 1.")
    if not df_presentations.index.equals(df_presentation_embeddings.index):
        raise ValueError("Indices of df_presentations and df_presentation_embeddings must match.")

    # Prepare presentation embeddings (numeric only)
    if COLUMNS['EMBEDDING_MODEL'] not in df_presentation_embeddings.columns:
        raise ValueError(f"'{COLUMNS['EMBEDDING_MODEL']}' column missing from df_presentation_embeddings.")
    model_name_presentations = df_presentation_embeddings[COLUMNS['EMBEDDING_MODEL']].iloc[0]
    # Record the mapping from DataFrame indices to array positions before any array operations
    pres_pos_to_idx, pres_idx_to_pos = create_index_mappings(df_presentations)

    # Convert embeddings DataFrame to numpy array for faster operations
    embeddings_array = df_presentation_embeddings.drop(columns=[COLUMNS['EMBEDDING_MODEL']]).values

    # Calculate similarity matrix for all general presentations
    similarity_matrix = similarity_func(embeddings_array, embeddings_array) # This is similarity within df_presentations
    # Convert similarity_matrix to numpy if it's a torch tensor
    if hasattr(similarity_matrix, 'cpu'):
        similarity_matrix = similarity_matrix.cpu().numpy()
    elif hasattr(similarity_matrix, 'numpy'):
        similarity_matrix = similarity_matrix.numpy()

    assigned_positions = set()  # Track which array positions are assigned
    final_clusters = []  # Will store array positions, convert to DF indices at the end
    hybrid_cluster_presentations = []  # Will store DataFrame indices of hybrid presentations in each cluster
    hybrid_session_titles = []  # Will store the session titles for hybrid sessions
        
    # Hybrid Sessions are optional, so check if they are provided
    if df_hybrid_presentations is not None and df_hybrid_embeddings is not None and hybrid_session_column is not None:
        if hybrid_session_column not in df_hybrid_presentations.columns:
            raise ValueError(f"hybrid_session_column '{hybrid_session_column}' must be present in df_hybrid_presentations.")   
        if not df_hybrid_presentations.index.equals(df_hybrid_embeddings.index):
            raise ValueError("Indices of df_hybrid_presentations and df_hybrid_embeddings must match.")
        if COLUMNS['EMBEDDING_MODEL'] not in df_hybrid_embeddings.columns:
            raise ValueError(f"'{COLUMNS['EMBEDDING_MODEL']}' column missing from df_hybrid_embeddings.")

        model_name_hybrid = df_hybrid_embeddings[COLUMNS['EMBEDDING_MODEL']].iloc[0]
        if model_name_hybrid != model_name_presentations:
            raise ValueError(
                f"Embedding model mismatch: Main presentations embedded with '{model_name_presentations}', "
                f"hybrid presentations with '{model_name_hybrid}'. They must use the same model."
            )
        
        numeric_hybrid_embeddings_df = df_hybrid_embeddings.drop(columns=[COLUMNS['EMBEDDING_MODEL']])
        
        # Create a representative embedding for each hybrid session
        # The resulting DataFrame will have hybrid session IDs (from hybrid_session_column) as its index.
        representative_hybrid_embeddings_df = numeric_hybrid_embeddings_df.groupby(
            df_hybrid_presentations[hybrid_session_column]
        ).mean()

        # Get count of presentations per session (same order as embeddings)
        session_counts = df_hybrid_presentations.groupby(hybrid_session_column).size()
        count_to_add = min_session_size - session_counts

        # Simarity matrix between hybrid sessions and general presentations
        # The resulting matrix will have shape (n_hybrid_sessions, n_presentations)
        hy_session_gen_pres_similarities = similarity_func(
            representative_hybrid_embeddings_df.values,
            embeddings_array
        )
        # Convert similarity_matrix to numpy if it's a torch tensor
        if hasattr(hy_session_gen_pres_similarities, 'cpu'):
            hy_session_gen_pres_similarities = hy_session_gen_pres_similarities.cpu().numpy()
        elif hasattr(hy_session_gen_pres_similarities, 'numpy'):
            hy_session_gen_pres_similarities = hy_session_gen_pres_similarities.numpy()

        presentations_to_hybrid = get_unique_top_indices_variable(hy_session_gen_pres_similarities, count_to_add)
        # Convert hybrid presentations to array positions and add to final clusters
        hybrid_session_names = list(representative_hybrid_embeddings_df.index)
        for session_idx, (session_name, additional_positions) in enumerate(zip(hybrid_session_names, presentations_to_hybrid)):
            # Get existing hybrid presentations for this session (convert DF indices to positions)
            existing_hybrid_df_indices = df_hybrid_presentations[
                df_hybrid_presentations[hybrid_session_column] == session_name
            ].index.tolist()
            final_clusters.append(list(additional_positions))
            hybrid_cluster_presentations.append(existing_hybrid_df_indices)
            hybrid_session_titles.append(session_name)  # Store the session title
            # Mark all these positions as assigned
            assigned_positions.update(list(additional_positions))
    
    # Create filtered embeddings and similarity matrix for unassigned presentations
    unassigned_positions = [i for i in range(len(embeddings_array)) if i not in assigned_positions]

    if len(unassigned_positions) == 0:
        # All presentations assigned to hybrid sessions
        pass
    else:
        # Create filtered arrays for clustering
        filtered_embeddings = embeddings_array[unassigned_positions]
        # filtered_similarity_matrix = similarity_matrix[np.ix_(unassigned_positions, unassigned_positions)]
        
        # Create mapping from filtered positions to original positions
        filtered_to_original = {i: unassigned_positions[i] for i in range(len(unassigned_positions))}
        
        # Perform hierarchical clustering on filtered data
        linkage_matrix = linkage(
            y=filtered_embeddings,
            method='average',
            metric='cosine',
        )
        
        # Clustering loop (similar to existing logic but with filtered data)
        n_nodes = linkage_matrix.shape[0]
        n_filtered = len(unassigned_positions)
        
        if n_nodes > 0:  # Only if there are presentations to cluster
            unassigned_count = linkage_matrix[:, 3].copy()
            unassigned_leaves = [[] for _ in range(n_nodes)]
            merge_stop_index = int(n_nodes * tree_merge_stop)
            
            for i in range(n_nodes):
                left_child, right_child = linkage_matrix[i, 0:2].astype(int)
                
                # Process children (using filtered positions)
                if left_child >= n_filtered:
                    left_idx = left_child - n_filtered
                    left_size = unassigned_count[left_idx]
                    unassigned_leaves[i].extend(unassigned_leaves[left_idx])
                else:
                    left_size = 1
                    unassigned_leaves[i].append(left_child)
                
                if right_child >= n_filtered:
                    right_idx = right_child - n_filtered
                    right_size = unassigned_count[right_idx]
                    unassigned_leaves[i].extend(unassigned_leaves[right_idx])
                else:
                    right_size = 1
                    unassigned_leaves[i].append(right_child)
                
                unassigned_count[i] = left_size + right_size
                
                # Check clustering conditions
                if (unassigned_count[i] >= min_session_size and 
                    i < merge_stop_index and 
                    len(final_clusters) < max_sessions):
                    
                    # Convert filtered positions back to original array positions
                    original_positions = [filtered_to_original[pos] for pos in unassigned_leaves[i]]
                    final_clusters.append(original_positions)
                    # For non-hybrid clusters, add empty list for hybrid presentations
                    hybrid_cluster_presentations.append([])
                    hybrid_session_titles.append(UNSET_SESSION_TITLE_TEXT)  # Non-hybrid sessions get default title
                    unassigned_count[i] = 0
                    unassigned_leaves[i] = []
            
            # Handle remaining unassigned items
            if unassigned_leaves[-1]:
                # Convert to original positions
                remaining_original = [filtered_to_original[pos] for pos in unassigned_leaves[-1]]
                final_clusters = _assign_remaining_items(remaining_original, final_clusters, similarity_matrix)
                # If we added any new clusters in _assign_remaining_items, we need to add empty hybrid lists and titles
                while len(hybrid_cluster_presentations) < len(final_clusters):
                    hybrid_cluster_presentations.append([])
                    hybrid_session_titles.append(UNSET_SESSION_TITLE_TEXT)

    # Convert all final_clusters from array positions to DataFrame indices
    final_clusters_df_indices = []
    for cluster_positions in final_clusters:
        cluster_df_indices = array_positions_to_df_indices(cluster_positions, pres_pos_to_idx)
        final_clusters_df_indices.append(cluster_df_indices)

    # Create outputs using DataFrame indices and pass hybrid information
    df_sessions, labels, metadata = _create_output_structures_with_df_indices(
        final_clusters_df_indices, df_presentations, cluster_column_name, 
        hybrid_cluster_presentations, hybrid_session_titles, final_session_title_column
    )

    return df_sessions, labels, metadata

def _create_output_structures_with_df_indices(final_clusters_df_indices, df_presentations, cluster_column_name, 
                                            hybrid_cluster_presentations=None, hybrid_session_titles=None, 
                                            final_session_title_column=COLUMNS['FINAL_SESSION_TITLE']):
    """
    Create output structures using DataFrame indices instead of array positions.
    """
    # Initialize cluster labels
    labels = pd.Series(-1, index=df_presentations.index, name=cluster_column_name)
    
    # Assign cluster labels
    for cluster_id, df_indices in enumerate(final_clusters_df_indices):
        labels.loc[df_indices] = cluster_id
    
    # Create sessions summary
    session_data = []
    for cluster_id, df_indices in enumerate(final_clusters_df_indices):
        # Get hybrid presentations for this cluster (if any)
        hybrid_presentations = []
        if hybrid_cluster_presentations and cluster_id < len(hybrid_cluster_presentations):
            hybrid_presentations = hybrid_cluster_presentations[cluster_id]
        
        # Get session title for this cluster
        session_title = UNSET_SESSION_TITLE_TEXT  # Default value
        if hybrid_session_titles and cluster_id < len(hybrid_session_titles):
            session_title = hybrid_session_titles[cluster_id]
        
        session_data.append({
            COLUMNS['CLUSTER_ID']: cluster_id,
            COLUMNS['SESSION_SIZE']: len(df_indices),
            COLUMNS['GEN_PRESENTATION_INDICES']: df_indices,
            COLUMNS['HYBRID_INVITED_PRESENTATIONS']: hybrid_presentations,
            final_session_title_column: session_title,
        })
    
    df_sessions = pd.DataFrame(session_data)
    
    # Create metadata
    n_assigned = sum(len(cluster) for cluster in final_clusters_df_indices)
    n_unassigned = len(df_presentations) - n_assigned
    
    metadata = {
        'n_clusters': len(final_clusters_df_indices),
        'n_assigned_items': n_assigned,
        'n_unassigned_items': n_unassigned,
        'n_total_items': len(df_presentations)
    }
    
    return df_sessions, labels, metadata


def _assign_remaining_items(remaining_positions, final_clusters, similarity_matrix):
    """
    Assign remaining items to existing clusters or create new ones.
    This version works with array positions.
    """
    if not remaining_positions or not final_clusters:
        if remaining_positions:
            final_clusters.append(remaining_positions)
        return final_clusters
    
    # For each remaining item, find the best cluster
    for pos in remaining_positions:
        best_cluster_idx = 0
        best_similarity = -1
        
        # Calculate average similarity to each existing cluster
        for cluster_idx, cluster_positions in enumerate(final_clusters):
            if cluster_positions:  # Only consider non-empty clusters
                similarities = [similarity_matrix[pos, cluster_pos] for cluster_pos in cluster_positions]
                avg_similarity = np.mean(similarities)
                
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_cluster_idx = cluster_idx
        
        # Add to best cluster
        final_clusters[best_cluster_idx].append(pos)
    
    return final_clusters

def calculate_placement_metrics(df_presentations, df_sessions, pres_similarities_matrix, session_column_name='Session Code'):
    """
    Calculate comprehensive quality metrics for session assignments.
    
    This function computes three key metrics to evaluate how well presentations 
    are organized into sessions:
    
    1. Session Coherence: Average pairwise similarity within each session 
        (measures internal session quality)
    2. Session Distinctiveness: Silhouette score for each session 
        (measures how unique/separable each session is from others)
    3. Presentation-Session Fit: For each presentation, average similarity 
        to other presentations in the same session (measures individual fit)

    Args:
        df_presentations (pd.DataFrame): DataFrame containing presentation data with 
            session assignments in the specified column.
        df_sessions (pd.DataFrame): DataFrame containing session metadata with 
            presentation indices stored in session_organizer.COLUMNS['GEN_PRESENTATION_INDICES'].
        pres_similarities_matrix (np.ndarray): Symmetric similarity matrix where 
            element [i,j] represents similarity between presentations i and j.
        session_column_name (str, optional): Column name in df_presentations that 
            contains session identifiers. Defaults to 'Session Code'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - df_sessions: Updated with 'session_coherence' and 'session_distinctiveness' columns
            - df_presentations: Updated with 'presentation_session_fit' column
            
    Raises:
        AssertionError: If presentation indices exceed the valid range for df_presentations.
        
    Note:
        The function modifies the input DataFrames in place and also returns them.
        Silhouette scores range from -1 to 1, where higher values indicate better separation.
        Similarity scores depend on the embedding model used but typically range from 0 to 1.
    """
    
    # Convert similarity to distance for silhouette calculation
    distance_matrix = 1 - pres_similarities_matrix
    np.fill_diagonal(distance_matrix, 0)

    pres_indices_by_session = df_sessions[COLUMNS['GEN_PRESENTATION_INDICES']]
    pos_to_ind, ind_to_pos= create_index_mappings(df_presentations)   
    pres_positions_by_session = pres_indices_by_session.apply(lambda indices_list: [ind_to_pos[i] for i in indices_list])
    max_position = max(chain.from_iterable(pres_positions_by_session))
    # Ensure max_position is within valid range
    assert max_position <= len(df_presentations) - 1, (
        f"Maximum position {max_position} exceeds maximum valid index {len(df_presentations) - 1} for df_presentations of length {len(df_presentations)}"
    )

    # Initialize outputs
    session_session_similarity_dataframe = pd.DataFrame(index=df_sessions[COLUMNS['CLUSTER_ID']], columns=df_sessions[COLUMNS['CLUSTER_ID']])
    session_average_similarity_series = pd.Series(index=df_sessions[COLUMNS['CLUSTER_ID']], dtype=float, name='session_coherence')
    presentation_session_fit_series = pd.Series(index=df_presentations.index, dtype=float, name='presentation_session_fit')


    # Calculate the silhouette scores for each presentation first
    # Get the assigned labels for each presentation
    labels = df_presentations[session_column_name].values
    sample_scores = silhouette_samples(distance_matrix, labels, metric='precomputed')

    # Create a pandas Series for session distinctiveness with matching index
    session_distinctiveness_series = pd.Series(index=df_sessions.index, dtype=float, name='session_distinctiveness')

    # Calculate average silhouette score for each cluster
    for _, row in df_sessions.iterrows():
        cluster_id = row[COLUMNS['CLUSTER_ID']]
        
        # Find samples belonging to this cluster in the assigned data
        cluster_mask = labels == cluster_id
        
        if np.sum(cluster_mask) > 0:
            cluster_score = np.mean(sample_scores[cluster_mask])
            session_distinctiveness_series.loc[row.name] = cluster_score

    # Calculate average similarity for each session
    for session_id_i, positions_i in pres_positions_by_session.items():
        for session_id_j, positions_j in pres_positions_by_session.items():
            if not positions_i or not positions_j:
                session_session_similarity_dataframe.loc[session_id_i, session_id_j] = 0.0
                continue
            session_i_to_j_similarities = pres_similarities_matrix[np.ix_(positions_i, positions_j)]
            if session_id_i == session_id_j:
                # First calculate the presentation to session similarities
                # Mask the diagonal (self-similarity) by setting it to np.nan, then compute mean of each row ignoring nan
                session_i_to_j_similarities_no_diag = session_i_to_j_similarities.copy()
                np.fill_diagonal(session_i_to_j_similarities_no_diag, np.nan)
                presentation_session_fits = np.nanmean(session_i_to_j_similarities_no_diag, axis=1)
                for i, pos in enumerate(positions_i):
                    presentation_session_fit_series.loc[pos_to_ind[pos]] = presentation_session_fits[i]
                # Same session: use upper triangle to avoid double-counting pairs
                n = session_i_to_j_similarities.shape[0]
                if n > 1:
                    triu_indices = np.triu_indices(n, k=1)
                    avg_similarity = np.mean(session_i_to_j_similarities[triu_indices])
                    # Store the average similarity in the series for this session
                    session_average_similarity_series.loc[session_id_i] = avg_similarity
                else:
                    avg_similarity = 0.0
            else:
                # Different sessions: use all similarities
                avg_similarity = np.mean(session_i_to_j_similarities)
            session_session_similarity_dataframe.loc[session_id_i, session_id_j] = avg_similarity
            
    return presentation_session_fit_series, session_average_similarity_series, session_distinctiveness_series, session_session_similarity_dataframe


# def calculate_avg_similarity(df_sessions, similarity_matrix):
#     """
#     Calculate average intra-cluster similarity for each cluster.
    
#     Args:
#         df_sessions (pd.DataFrame): DataFrame with cluster assignments
#         similarity_matrix (np.ndarray): Similarity matrix of presentations
        
#     Returns:
#         list: Average similarity scores for each cluster
#     """
#     avg_similarities = []
    
#     for _, row in df_sessions.iterrows():
#         cluster_indices = row[COLUMNS['GEN_PRESENTATION_INDICES']]
        
#         if len(cluster_indices) < 2:
#             # Single item clusters have no internal similarity
#             avg_similarities.append(np.nan)
#             continue
        
#         # Get all pairwise similarities within the cluster
#         cluster_similarities = []
#         for i in range(len(cluster_indices)):
#             for j in range(i + 1, len(cluster_indices)):
#                 idx_i = cluster_indices[i]
#                 idx_j = cluster_indices[j]
#                 cluster_similarities.append(similarity_matrix[idx_i, idx_j])
        
#         # Calculate average similarity
#         avg_similarity = np.mean(cluster_similarities)
#         avg_similarities.append(avg_similarity)
    
#     return avg_similarities

# def calculate_silhouette_scores(df_sessions, similarity_array, labels, ):
#     """
#     Calculate silhouette scores using the same similarity function as the embedding model.
    
#     Args:
#         df_sessions (pd.DataFrame): DataFrame with cluster assignments
#         similarity_array (np.ndarray): Precomputed similarity matrix
#         labels (list): List of cluster labels for each presentation


#     Returns:
#         list: Silhouette scores for each cluster
#     """
#     from sklearn.metrics import silhouette_samples
    
#     # Convert labels to numpy array and handle unassigned items
#     labels_array = np.array(labels)

#     # Convert similarity to distance: distance = 1 - similarity
#     # Ensure diagonal is exactly 0 for numerical stability
#     distance_matrix = 1 - similarity_array
#     np.fill_diagonal(distance_matrix, 0)
    
#     # Only calculate silhouette for assigned items (exclude -1 labels)
#     assigned_mask = labels_array != -1
    
#     if np.sum(assigned_mask) < 2:
#         return [np.nan] * len(df_sessions)
    
#     # Filter distance matrix and labels for assigned items
#     distance_matrix = distance_matrix[assigned_mask][:, assigned_mask]
#     assigned_labels = labels_array[assigned_mask]
    
#     # Check if we have at least 2 different clusters
#     unique_labels = np.unique(assigned_labels)
#     if len(unique_labels) < 2:
#         return [np.nan] * len(df_sessions)
       
#     # Calculate silhouette scores for each sample
#     sample_scores = silhouette_samples(distance_matrix, assigned_labels, metric='precomputed')
    
#     # Calculate average silhouette score for each cluster
#     cluster_silhouette_scores = []
    
#     for _, row in df_sessions.iterrows():
#         cluster_id = row[COLUMNS['CLUSTER_ID']]
        
#         # Find samples belonging to this cluster in the assigned data
#         cluster_mask = assigned_labels == cluster_id
        
#         if np.sum(cluster_mask) > 0:
#             cluster_score = np.mean(sample_scores[cluster_mask])
#             cluster_silhouette_scores.append(cluster_score)
#         else:
#             cluster_silhouette_scores.append(np.nan)
    
#     return cluster_silhouette_scores

# def calculate_document_similarities(similarity_matrix, labels):
#     """
    
#     Calculate average similarity of each document to others in its cluster
#     This version has been vectorized for better performance with large datasets
#     Args:
#         similarity_matrix (np.ndarray): Similarity matrix of shape (n_samples, n_samples)
#         labels (array-like): Cluster labels for each document
    
#     Returns:
#         np.ndarray: Average similarity of each document to its cluster
#     """
#         # Ensure labels is a numpy array for proper vectorized operations
#     labels = np.array(labels)
    
#     # Input validation
#     n_samples = len(labels)
#     if similarity_matrix.shape[0] != n_samples or similarity_matrix.shape[1] != n_samples:
#         raise ValueError(f"Similarity matrix shape {similarity_matrix.shape} doesn't match labels length {n_samples}")
#     document_similarities = np.zeros(n_samples)
    
#     # Process each unique cluster
#     unique_labels = np.unique(labels)
#     unique_labels = unique_labels[unique_labels != -1]  # Exclude unassigned
    
#     for cluster_label in unique_labels:
#         cluster_mask = labels == cluster_label
#         cluster_indices = np.where(cluster_mask)[0]
        
#         if len(cluster_indices) > 1:  # Only process multi-item clusters
#             # Extract submatrix for this cluster
#             cluster_sim_matrix = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
            
#             # Calculate mean similarity for each item (excluding diagonal)
#             cluster_means = (cluster_sim_matrix.sum(axis=1) - np.diag(cluster_sim_matrix)) / (len(cluster_indices) - 1)
            
#             # Assign back to main array
#             document_similarities[cluster_indices] = cluster_means
    
#     return document_similarities

# Default prompts as module-level constants
DEFAULT_SESSION_PROMPT = """I am organizing oral research presentation sessions for the American Society of Biological and Agricultural Engineers Annual International Meeting. Please provide 3 options for the name/title of a session. Also provide 5 keywords describing the session. The name and keywords should highlight the commonality among all presentations. The target audience for titles and keywords is engineering designers and researchers. The title should be descriptive of the content and be interesting and engaging. It should be less than 100 characters long.

Please respond in JSON format:
{{"title1": "Session Title", "title2": "Session Title", "title3": "Session Title", "keywords": "keyword1, keyword2, keyword3, keyword4, keyword5"}}

The titles and abstracts for presentations assigned to this session are:
{presentations}"""

DEFAULT_GEMINI_SESSION_PROMPT = """I am organizing oral research presentation sessions for the American Society of Biological and Agricultural Engineers Annual International Meeting. Please provide 3 options for the name/title of a session. Also provide 5 keywords describing the session. The name and keywords should highlight the commonality among all presentations. The target audience for titles and keywords is engineering designers and researchers. The title should be descriptive of the content and be interesting and engaging. It should be less than 100 characters long.

Please respond in this format:

{{
     'title_1': 'Session Title',
     'title_2': 'Session Title',
     'title_3': 'Session Title',
     'keywords': 'keyword1, keyword2, keyword3, keyword4, keyword5'
}}

The titles and abstracts for presentations assigned to this session are:
{presentations}"""

def generate_session_titles_and_keywords_ollama(df_sessions, df_presentations, topic_column="Title and Abstract", model_name="llama3.2:3b", prompt_template=None):
    """
    Generate session titles and keywords using Ollama with long context and JSON schema.
    
    Args:
        df_sessions (pd.DataFrame): DataFrame with session information
        df_presentations (pd.DataFrame): DataFrame with presentation data
        topic_column (str): Column name containing the combined title and abstract text
        model_name (str): Ollama model name (e.g., "llama3.2:3b", "mistral:7b")
        prompt_template (str): Custom prompt template with {presentations} placeholder
        
    Returns:
        pd.DataFrame: df_sessions with added columns for generated titles and keywords
    """
    import requests
    import json
    import time
    
    # Use default prompt if none provided
    if prompt_template is None:
        prompt_template = DEFAULT_SESSION_PROMPT
    
    # JSON schema for structured output
    json_schema = {
        "type": "object",
        "properties": {
            "title1": {
                "type": "string",
                "description": "First session title option"
            },
            "title2": {
                "type": "string", 
                "description": "Second session title option"
            },
            "title3": {
                "type": "string",
                "description": "Third session title option"
            },
            "keywords": {
                "type": "string",
                "description": "Five comma-separated keywords"
            }
        },
        "required": ["title1", "title2", "title3", "keywords"],
        "additionalProperties": False
    }
    
    # Check if Ollama is running and model is available
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            raise ConnectionError("Ollama server not responding")
        
        available_models = [model["name"] for model in response.json()["models"]]
        if model_name not in available_models:
            print(f"Model '{model_name}' not found. Available models: {available_models}")
            print(f"Run 'ollama pull {model_name}' to download it.")
            raise ValueError(f"Model {model_name} not available")
            
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Cannot connect to Ollama server: {e}. Please run 'ollama serve' first.")
    
    print(f"Using model: {model_name}")
    
    # Create sessions dictionary from df_sessions
    sessions_dict = {}
    for _, row in df_sessions.iterrows():
        session_num = row[COLUMNS['CLUSTER_ID']]
        presentation_indices = row[COLUMNS['GEN_PRESENTATION_INDICES']]
        
        sessions_dict[session_num] = {
            "Indices": presentation_indices,
            "Presentations": [],
        }
        
        for index in presentation_indices:
            try:
                title_abstract = df_presentations.loc[index, topic_column]
                sessions_dict[session_num]["Presentations"].append(title_abstract)
            except KeyError:
                print(f"Warning: Index {index} not found in df_presentations for session {session_num}. Skipping.")
                continue
    
    start_time = time.time()
    total_sessions = len(sessions_dict)
    
    # Process each session
    for idx, (session_key, session_value) in enumerate(sessions_dict.items(), 1):
        print(f"Processing session {session_key} ({idx}/{total_sessions})...")
        
        # Format the prompt with presentations
        prompt = prompt_template.format(presentations=str(session_value["Presentations"]))
        
        try:
            # Make request to Ollama with extended context and schema
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "format": json_schema,  # Use JSON schema for structured output
                    "stream": False,
                    "options": {
                        "num_ctx": 10000,  # Use extended context length
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                    }
                },
                timeout=120  # Longer timeout for extended context
            )
            
            if response.status_code == 200:
                result = response.json()
                text_answer = result["response"].strip()
                
                try:
                    # Parse JSON response
                    answer_dict = json.loads(text_answer)
                    
                    # Store results with validation
                    session_value[f"Ollama: {model_name} Title 1"] = answer_dict.get("title1", f"No Title Generated")
                    session_value[f"Ollama: {model_name} Title 2"] = answer_dict.get("title2", f"No Title Generated")
                    session_value[f"Ollama: {model_name} Title 3"] = answer_dict.get("title3", f"No Title Generated")
                    session_value[f"Ollama: {model_name} Keywords"] = answer_dict.get("keywords", "No Keywords Generated")

                    print(f"  ✓ Generated titles for session {session_key}")
                    
                except json.JSONDecodeError as e:
                    print(f"  ✗ JSON parsing failed for session {session_key}: {e}")
                    print(f"  Raw response: {text_answer[:1000]}...")
                    
                    # Set fallback values
                    session_value[f"Ollama: {model_name} Title 1"] = f"No Title Generated"
                    session_value[f"Ollama: {model_name} Title 2"] = f"No Title Generated"
                    session_value[f"Ollama: {model_name} Title 3"] = f"No Title Generated"
                    session_value[f"Ollama: {model_name} Keywords"] = "No Keywords Generated"

            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"  ✗ Error processing session {session_key}: {e}")
            # Set fallback values
            session_value[f"Ollama: {model_name} Title 1"] = f"No Title Generated"
            session_value[f"Ollama: {model_name} Title 2"] = f"No Title Generated"
            session_value[f"Ollama: {model_name} Title 3"] = f"No Title Generated"
            session_value[f"Ollama: {model_name} Keywords"] = "No Keywords Generated"

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per session: {elapsed_time/total_sessions:.2f} seconds")
    
    # Add generated content back to df_sessions
    df_sessions_with_titles = df_sessions.copy()
    
    # Create lists for new columns
    title_1_list = []
    title_2_list = []
    title_3_list = []
    keywords_list = []
    
    for _, row in df_sessions_with_titles.iterrows():
        session_id = row[COLUMNS['CLUSTER_ID']]
        if session_id in sessions_dict:
            title_1_list.append(sessions_dict[session_id].get(f"Ollama: {model_name} Title 1", "No Title Generated"))
            title_2_list.append(sessions_dict[session_id].get(f"Ollama: {model_name} Title 2", "No Title Generated"))
            title_3_list.append(sessions_dict[session_id].get(f"Ollama: {model_name} Title 3", "No Title Generated"))
            keywords_list.append(sessions_dict[session_id].get(f"Ollama: {model_name} Keywords", "No Keywords Generated"))
        else:
            title_1_list.append("No Title Generated")
            title_2_list.append("No Title Generated")
            title_3_list.append("No Title Generated")
            keywords_list.append("No Keywords Generated")
    
    # Add new columns to DataFrame
    df_sessions_with_titles[f"Ollama: {model_name} Title 1"] = title_1_list
    df_sessions_with_titles[f"Ollama: {model_name} Title 2"] = title_2_list
    df_sessions_with_titles[f"Ollama: {model_name} Title 3"] = title_3_list
    df_sessions_with_titles[f"Ollama: {model_name} Keywords"] = keywords_list

    return df_sessions_with_titles



class Session_info(BaseModel):
    title_1: str
    title_2: str
    title_3: str
    keywords: str

def generate_session_titles_and_keywords_gemini(
    df_sessions,
    df_presentations,
    topic_column="Title and Abstract",
    prompt_template=None,
    api_key=None,
):
    """
    Generate session titles and keywords using Google's Gemini API.

    Args:
        df_sessions (pd.DataFrame): DataFrame with session information from create_sessions()
        df_presentations (pd.DataFrame): DataFrame with presentation data
        topic_column (str): Column name containing the combined title and abstract text
        prompt_template (str): Custom prompt template with {presentations} placeholder
        api_key (str): Gemini API key. If not provided, it will be loaded from environment variables.

    Returns:
        pd.DataFrame: df_sessions with added columns for generated titles and keywords
    """
    # Load environment variables
    load_dotenv(".env")

    # Use default prompt if none provided
    if prompt_template is None:
        prompt_template = DEFAULT_GEMINI_SESSION_PROMPT

    # Check for API key in environment variables if not provided
    if api_key is None:
        if "GEMINI_API_KEY" not in os.environ:
            raise ValueError(
                "API key must be provided or in environmental variables. GEMINI_API_KEY not found in environment variables. Please set it in your .env file."
            )
        else:
            api_key = os.environ["GEMINI_API_KEY"]

    # Validate API key
    if not api_key:
        raise ValueError("API key is required to use Gemini API.")

    # Create sessions dictionary from df_sessions
    sessions_dict = {}
    for _, row in df_sessions.iterrows():
        session_num = row[COLUMNS['CLUSTER_ID']]
        presentation_indices = row[COLUMNS['GEN_PRESENTATION_INDICES']]

        sessions_dict[session_num] = {
            "Indices": presentation_indices,
            "Presentations": [],
        }

        for index in presentation_indices:
            try:
                title_abstract = df_presentations.loc[index, topic_column]
                sessions_dict[session_num]["Presentations"].append(title_abstract)
            except KeyError:
                print(
                    f"Warning: Index {index} not found in df_presentations for session {session_num}. Skipping."
                )
                continue

    start_time = time.time()
    total_sessions = len(sessions_dict)

    # Create the client
    client = genai.Client(api_key=api_key)

    # Generate titles and keywords for each session
    for idx, (session_key, session_value) in enumerate(sessions_dict.items(), 1):
        print(f"Processing session {session_key} ({idx}/{total_sessions})...")

        try:
            # Format the prompt with the presentations
            formatted_prompt = prompt_template.format(
                presentations=str(session_value["Presentations"])
            )

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=formatted_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Session_info,
                },
            )

            # Parse the response
            my_sessions: Session_info = response.parsed
            print(f"  ✓ Generated titles for session {session_key}")

            # Store results in sessions_dict
            session_value["Gemini Title 1"] = my_sessions.title_1
            session_value["Gemini Title 2"] = my_sessions.title_2
            session_value["Gemini Title 3"] = my_sessions.title_3
            session_value["Gemini Keywords"] = my_sessions.keywords

            # Add small delay to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"  ✗ Error processing session {session_key}: {e}")
            # Set default values in case of error
            session_value["Gemini Title 1"] = f"No Title Generated"
            session_value["Gemini Title 2"] = f"No Title Generated"
            session_value["Gemini Title 3"] = f"No Title Generated"
            session_value["Gemini Keywords"] = (
                "No Keywords Generated"
            )

    end_time = time.time()
    gemini_elapsed_time = end_time - start_time
    print(f"\nTotal processing time: {gemini_elapsed_time:.4f} seconds")
    print(f"Average time per session: {gemini_elapsed_time/total_sessions:.4f} seconds")

    # Add the generated content back to df_sessions
    df_sessions_with_titles = df_sessions.copy()

    # Create lists to store the generated content in the same order as df_sessions
    title_1_list = []
    title_2_list = []
    title_3_list = []
    keywords_list = []

    for _, row in df_sessions_with_titles.iterrows():
        session_id = row[COLUMNS['CLUSTER_ID']]
        if session_id in sessions_dict:
            title_1_list.append(
                sessions_dict[session_id].get("Gemini Title 1", "No Title Generated")
            )
            title_2_list.append(
                sessions_dict[session_id].get("Gemini Title 2", "No Title Generated")
            )
            title_3_list.append(
                sessions_dict[session_id].get("Gemini Title 3", "No Title Generated")
            )
            keywords_list.append(
                sessions_dict[session_id].get(
                    "Gemini Keywords", "No Keywords Generated"
                )
            )
        else:
            title_1_list.append("No Title Generated")
            title_2_list.append("No Title Generated")
            title_3_list.append("No Title Generated")
            keywords_list.append("No Keywords Generated")

    # Add the new columns to the DataFrame
    df_sessions_with_titles["Gemini Title 1"] = title_1_list
    df_sessions_with_titles["Gemini Title 2"] = title_2_list
    df_sessions_with_titles["Gemini Title 3"] = title_3_list
    df_sessions_with_titles["Gemini Keywords"] = keywords_list

    return df_sessions_with_titles


def generate_session_titles_and_keywords_llama_local(
    df_sessions,
    df_presentations,
    topic_column="Title and Abstract",
    prompt_template=None,
):
    """
    Generate session titles and keywords using LLaMA locally via llama-cpp-python.

    Args:
        df_sessions (pd.DataFrame): DataFrame with session information from create_sessions()
        df_presentations (pd.DataFrame): DataFrame with presentation data
        topic_column (str): Column name containing the combined title and abstract text
        prompt_template (str): Custom prompt template with {presentations} placeholder

    Returns:
        pd.DataFrame: df_sessions with added columns for generated titles and keywords
    """
    import llama_cpp
    import json
    import time

    # Use default prompt if none provided
    if prompt_template is None:
        prompt_template = DEFAULT_SESSION_PROMPT

    # Initialize model with updated parameters
    try:
        model = llama_cpp.Llama(
            model_path="llama-3.2-3b-instruct-q8_0.gguf",
            n_ctx=0,  # Do not set a specific context size
            n_gpu_layers=-1,  # Use all available GPU layers (if any) or CPU (if no GPU is available)
            verbose=False,  # Reduce verbose output
            chat_format="llama-3",  # Specify chat format for LLaMA 3.x
            n_threads=None,  # Auto-detect optimal thread count
        )
        print("LLaMA model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load LLaMA model: {e}")

    # Create sessions dictionary from df_sessions
    sessions_dict = {}
    for _, row in df_sessions.iterrows():
        session_num = row[COLUMNS['CLUSTER_ID']]
        presentation_indices = row[COLUMNS['GEN_PRESENTATION_INDICES']]

        sessions_dict[session_num] = {
            "Indices": presentation_indices,
            "Presentations": [],
        }

        for index in presentation_indices:
            try:
                title_abstract = df_presentations.loc[index, topic_column]
                sessions_dict[session_num]["Presentations"].append(title_abstract)
            except KeyError:
                print(
                    f"Warning: Index {index} not found in df_presentations for session {session_num}. Skipping."
                )
                continue

    start_time = time.time()
    total_sessions = len(sessions_dict)

    # Iterate over the sessions and prompt the LLM for the session title and keywords
    for idx, (session_key, session_value) in enumerate(sessions_dict.items(), 1):
        print(f"Processing session {session_key} ({idx}/{total_sessions})...")

        # Format the prompt with the presentations
        formatted_prompt = prompt_template.format(
            presentations=str(session_value["Presentations"])
        )

        messages = [{"role": "user", "content": formatted_prompt}]

        try:
            session_titleAbsLlama = model.create_chat_completion(
                messages=messages,
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title1": {"type": "string"},
                            "title2": {"type": "string"},
                            "title3": {"type": "string"},
                            "keywords": {"type": "string"},
                        },
                        "required": ["title1", "title2", "title3", "keywords"],
                    },
                },
                temperature=0.7,  # Add temperature control
                max_tokens=500,  # Limit response length
                top_p=0.9,  # Add top_p sampling
            )

            text_answer = session_titleAbsLlama["choices"][0]["message"]["content"]

            try:
                # Parse the JSON string into a Python dictionary
                answer_dict = json.loads(text_answer)

                # Access the values using the dictionary keys
                title_1 = answer_dict.get("title1", f"No Title Generated")
                title_2 = answer_dict.get("title2", f"No Title Generated")
                title_3 = answer_dict.get("title3", f"No Title Generated")
                keywords = answer_dict.get(
                    "keywords",
                    "No Keywords Generated",
                )

                # Now you can save the titles and keywords
                session_value["Llama Title 1"] = title_1
                session_value["Llama Title 2"] = title_2
                session_value["Llama Title 3"] = title_3
                session_value["Llama Keywords"] = keywords

                print(f"  ✓ Generated titles for session {session_key}")

            except json.JSONDecodeError as e:
                print(f"  ✗ Error decoding JSON for session {session_key}: {e}")
                print(f"Problematic JSON string: {text_answer}")
                # Handle the error appropriately, e.g., set default values or skip the session
                session_value["Llama Title 1"] = "Error: Could not generate title"
                session_value["Llama Title 2"] = "Error: Could not generate title"
                session_value["Llama Title 3"] = "Error: Could not generate title"
                session_value["Llama Keywords"] = "Error: Could not generate keywords"

        except Exception as e:
            print(f"  ✗ Error processing session {session_key}: {e}")
            session_value["Llama Title 1"] = f"No Title Generated"
            session_value["Llama Title 2"] = f"No Title Generated"
            session_value["Llama Title 3"] = f"No Title Generated"
            session_value["Llama Keywords"] = (
                "No Keywords Generated"
            )

    end_time = time.time()
    llama_elapsed_time = end_time - start_time
    print(f"\nTotal processing time: {llama_elapsed_time:.4f} seconds")
    print(f"Average time per session: {llama_elapsed_time/total_sessions:.4f} seconds")

    # Add the generated content back to df_sessions
    df_sessions_with_titles = df_sessions.copy()

    # Create lists to store the generated content in the same order as df_sessions
    title_1_list = []
    title_2_list = []
    title_3_list = []
    keywords_list = []

    for _, row in df_sessions_with_titles.iterrows():
        session_id = row[COLUMNS['CLUSTER_ID']]
        if session_id in sessions_dict:
            title_1_list.append(
                sessions_dict[session_id].get("Llama Title 1", "No Title Generated")
            )
            title_2_list.append(
                sessions_dict[session_id].get("Llama Title 2", "No Title Generated")
            )
            title_3_list.append(
                sessions_dict[session_id].get("Llama Title 3", "No Title Generated")
            )
            keywords_list.append(
                sessions_dict[session_id].get("Llama Keywords", "No Keywords Generated")
            )
        else:
            title_1_list.append("No Title Generated")
            title_2_list.append("No Title Generated")
            title_3_list.append("No Title Generated")
            keywords_list.append("No Keywords Generated")

    # Add the new columns to the DataFrame
    df_sessions_with_titles["Llama Title 1"] = title_1_list
    df_sessions_with_titles["Llama Title 2"] = title_2_list
    df_sessions_with_titles["Llama Title 3"] = title_3_list
    df_sessions_with_titles["Llama Keywords"] = keywords_list

    return df_sessions_with_titles


def generate_session_titles_and_keywords(
    df_sessions,
    df_presentations,
    topic_column="Title and Abstract",
    model_name="gemini-2.0-flash",
    prompt_template=None,
    api_key=None,
):
    """
    Generate session titles and keywords using various AI models.

    Args:
        df_sessions: DataFrame with session information
        df_presentations: DataFrame with presentation data
        topic_column: Column name containing presentation text
        model_name: Model to use ('gemini-2.0-flash', 'llama-3.2-local', 'ollama:model_name')
        prompt_template: Custom prompt template with {presentations} placeholder
        api_key: API key for Gemini if using that model
    Returns:
        pd.DataFrame: df_sessions with added columns for generated titles and keywords
    """
    if model_name == "gemini-2.0-flash":
        return generate_session_titles_and_keywords_gemini(
            df_sessions, df_presentations, topic_column, prompt_template, api_key
        )
    elif model_name == "llama-3.2-local":
        return generate_session_titles_and_keywords_llama_local(
            df_sessions, df_presentations, topic_column, prompt_template
        )
    elif model_name.startswith("ollama:"):
        # Extract model name after 'ollama:'
        ollama_model = model_name.split(":", 1)[1]
        return generate_session_titles_and_keywords_ollama(
            df_sessions, df_presentations, topic_column, ollama_model, prompt_template
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def find_most_similar_committees_by_presentations(
    df_sessions,
    df_presentation_embeddings,
    df_committees,
    committee_embeddings,
    top_n=3,
):
    """Improved version with better error handling and performance."""

    # Input validation
    if df_sessions.empty:
        return pd.DataFrame()

    if hasattr(committee_embeddings, "values"):
        if 'embedding_model' in committee_embeddings.columns:
            committee_embeddings = committee_embeddings.drop(columns=['embedding_model'])
        committee_embeddings = committee_embeddings.values

    if 'embedding_model' in df_presentation_embeddings.columns:
                    df_presentation_embeddings = df_presentation_embeddings.drop(columns=['embedding_model'])
            
    # Pre-normalize committee embeddings once
    committee_norms = committee_embeddings / np.linalg.norm(
        committee_embeddings, axis=1, keepdims=True
    )
    results = []

    for _, cluster_row in df_sessions.iterrows():
        session_id = cluster_row[COLUMNS['CLUSTER_ID']]
        presentation_indices = cluster_row[COLUMNS['GEN_PRESENTATION_INDICES']]
        print(f"Processing session {session_id} with {len(presentation_indices)} presentations...")

        if len(presentation_indices) == 0:
            continue
        try:
            # Vectorized operations
            session_embeddings = df_presentation_embeddings.iloc[
                presentation_indices
            ].values
            session_avg_embedding = session_embeddings.mean(axis=0)
            session_norm = session_avg_embedding / np.linalg.norm(session_avg_embedding)

            # Calculate all similarities at once
            similarities = np.dot(committee_norms, session_norm)
            top_indices = similarities.argsort()[-top_n:][::-1]

            # Batch create results for this session
            session_results = [
                {
                    "Clustering Session": session_id,
                    "Committee_Rank": rank,
                    "Committee_Name": df_committees.iloc[committee_idx][
                        "Committee_Name"
                    ],
                    "Committee_Description": df_committees.iloc[committee_idx][
                        "Description"
                    ],
                    "Similarity_Score": similarities[committee_idx],
                    "Session_Size": len(presentation_indices),
                }
                for rank, committee_idx in enumerate(top_indices, 1)
            ]
            results.extend(session_results)

        except (IndexError, KeyError) as e:
            print(f"Warning: Error processing session {session_id}: {e}")
            continue

    return pd.DataFrame(results)


# More efficient way to add committee information using merge
def add_committee_matches_to_clusters(df_sessions, session_committee_matches):
    """Add committee match information to clusters DataFrame efficiently."""

    # Pivot the matches to get one row per session
    committee_pivot = session_committee_matches.pivot(
        index="Clustering Session",
        columns="Committee_Rank",
        values=["Committee_Name", "Similarity_Score"],
    )

    # Flatten column names
    committee_pivot.columns = [
        f"{col[0].replace('Committee_', '')}{col[1]}" for col in committee_pivot.columns
    ]

    # Rename for clarity
    rename_map = {
        "Name1": "Top Committee Match",
        "Similarity_Score1": "Top Committee Similarity",
        "Name2": "2nd Committee Match",
        "Similarity_Score2": "2nd Committee Similarity",
        "Name3": "3rd Committee Match",
        "Similarity_Score3": "3rd Committee Similarity",
    }
    committee_pivot = committee_pivot.rename(columns=rename_map)

    # Merge with clusters
    return df_sessions.merge(
        committee_pivot, left_on=COLUMNS['CLUSTER_ID'], right_index=True, how="left"
    )

def load_hybrid_sessions(file_path, Session_column='Session', Title_column='Title', Abstract_column='Abstract', 
                        Abstract_ID_column='Submission ID - 7 digits',
                        session_column='Session', title_column='Title', abstract_column='Abstract', 
                        abstract_id_column='Abstract ID', topic_column='Title and Abstract'):
    """
    Load hybrid session presentations from a CSV or Excel file and create both presentations and sessions DataFrames.
    
    Args:
        file_path (str): Path to the CSV or Excel file containing hybrid session information
        Session_column (str): Spreadsheet column name that contains the session names
        Title_column (str): Spreadsheet column name that contains the presentation titles
        Abstract_column (str): Spreadsheet column name that contains the abstracts
        Abstract_ID_column (str): Spreadsheet column name that contains the abstract IDs
        session_column (str): Name for the session column in the output DataFrame
        title_column (str): Name for the title column in the output DataFrame
        abstract_column (str): Name for the abstract column in the output DataFrame
        abstract_id_column (str): Name for the abstract ID column in the output DataFrame
        topic_column (str): Name for the combined title and abstract column in the output DataFrame
        
    Returns:
        tuple: (df_presentations, df_sessions, session_column, title_column, abstract_column, abstract_id_column, topic_column)
    """
    # Determine file type and read accordingly
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format. Please use CSV (.csv) or Excel (.xlsx, .xls) files.")
    
    # Validate required columns exist in the file
    required_columns = [Session_column, Title_column, Abstract_column, Abstract_ID_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. "
                        f"Expected columns: {required_columns}")
    
    # Create presentations DataFrame
    # First, rename the key columns
    column_rename_map = {
        Session_column: session_column,
        Title_column: title_column, 
        Abstract_column: abstract_column,
        Abstract_ID_column: abstract_id_column
    }
    
    df_presentations = df.rename(columns=column_rename_map)
    
    # Clean up any NaN values in critical columns
    df_presentations = df_presentations.dropna(subset=[session_column, title_column, abstract_column])
    
    # Create combined topic column (like load_presentations)
    df_presentations[topic_column] = df_presentations[[title_column, abstract_column]].agg(': '.join, axis=1)
    
    # Reset index to ensure clean sequential indexing
    df_presentations = df_presentations.reset_index(drop=True)
    
    # Create sessions DataFrame (like create_sessions output)
    session_data = []
    
    # Create mapping from session names to integer IDs (starting from 1)
    unique_sessions = sorted(df_presentations[session_column].unique())
    session_name_to_id = {name: idx + 1 for idx, name in enumerate(unique_sessions)}
    
    # Group by session to create session information
    for session_name, group in df_presentations.groupby(session_column):
        presentation_indices = group.index.tolist()
        cluster_id = session_name_to_id[session_name]
        
        session_data.append({
            COLUMNS['CLUSTER_ID']: cluster_id,  # Use integer ID starting from 1
            COLUMNS['GEN_PRESENTATION_INDICES']: presentation_indices,
            COLUMNS['SESSION_SIZE']: len(presentation_indices),
            COLUMNS['HYBRID_SESSION_TITLE']: session_name  # Keep original session name as title
        })
    
    # Sort by cluster_id to ensure consistent ordering
    session_data = sorted(session_data, key=lambda x: x[COLUMNS['CLUSTER_ID']])
    df_sessions = pd.DataFrame(session_data)
    
    # Add cluster_id mapping to presentations DataFrame for consistency
    df_presentations[COLUMNS['CLUSTER_ID']] = df_presentations[session_column].map(session_name_to_id)
    
    print(f"Loaded {len(df_presentations)} hybrid presentations in {len(df_sessions)} sessions")
    print(f"Session mapping: {session_name_to_id}")
    
    return df_presentations, df_sessions, session_column, title_column, abstract_column, abstract_id_column, topic_column


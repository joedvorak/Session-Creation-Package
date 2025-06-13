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
        pd.DataFrame: DataFrame containing the prepared presentation information.
    """
    df = pd.read_excel(file_path)
    if Title_name not in df.columns or Abstract_name not in df.columns or Abstract_ID_name not in df.columns:
        raise ValueError(f"Columns '{Title_name}', '{Abstract_name}', '{Abstract_ID_name}' must be present in the Excel file.")
    df = df[[Title_name, Abstract_name, Abstract_ID_name]].rename(columns={Title_name: title_column, Abstract_name: abstract_column, Abstract_ID_name: abstract_id_column})

    # Drop any presentations that are missing an abstrct or title
    df.dropna(subset=[title_column,abstract_column], inplace=True)
    
    # Combine Titles and Abstracts with a semicolon in between. 
    # This should be the same as + but .agg() handles empty fields or fields that have non-text entries.
    df[topic_column] = df[[title_column, abstract_column,]].agg(': '.join, axis=1)
    
    return df, title_column, abstract_column, abstract_id_column, topic_column

import pandas as pd

def parse_committee_file_simple(file_path):
    """
    Parse a committee file where each committee follows the pattern:
    - Committee name (single line)
    - Committee description (one or more lines)
    - Blank line
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    committees = []
    current_committee = None
    current_description_lines = []
    
    for line in lines:
        line = line.strip()
        
        if not line:  # Blank line - end of current committee
            if current_committee is not None:
                description = ' '.join(current_description_lines).strip()
                committees.append({
                    'Committee_Name': current_committee,
                    'Description': description,
                    'Name_Description': f"{current_committee}: {description}"
                })
                current_committee = None
                current_description_lines = []
        
        elif current_committee is None:  # First non-blank line after blank/start = committee name
            current_committee = line
            current_description_lines = []
        
        else:  # Continuation of description
            current_description_lines.append(line)
    
    # Handle the last committee if file doesn't end with blank line
    if current_committee is not None:
        description = ' '.join(current_description_lines).strip()
        committees.append({
            'Committee_Name': current_committee,
            'Description': description,
            'Name_Description': f"{current_committee}: {description}"
        })
    return pd.DataFrame(committees)

def embed_documents(df_presentations, topic_column, embedding_model):
    if not isinstance(embedding_model, SentenceTransformer):
        raise ValueError("embedding_model must be an instance of SentenceTransformer.")
    if topic_column not in df_presentations.columns:
        raise ValueError(f"topic_column '{topic_column}' must be present in the DataFrame.")
    if embedding_model.model_card_data.base_model == "jxm/cde-small-v1":
        # This is the CDE model, so we need to use the special CDE embedding function
        return cde_embed_documents(df_presentations, topic_column, embedding_model)
    else:
        # This is a standard SentenceTransformer model, so we can use the standard embedding function
        return standard_embed_documents(df_presentations, topic_column, embedding_model)

def standard_embed_documents(df_presentations, topic_column, embedding_model):
    """
    Embed the presentation topics using a standard SentenceTransformer model.
    
    Args:
        df_presentations (pd.DataFrame): DataFrame containing the presentations.
        topic_column (str): Column name in df_presentations that contains the topics to embed.
        embedding_model (SentenceTransformer): The SentenceTransformer model to use for embedding.
        
    Returns:
        pd.DataFrame: DataFrame containing the embeddings of the presentation topics.
        pd.DataFrame: DataFrame containing the similarity matrix of the presentation embeddings.
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
    
    return df_presentation_embeddings

def cde_embed_documents(df_presentations, topic_column, cde_embeddings_model):
    """
    Embed the presentation topics using a CDE (Contextual Document Embeddings) model.
    Args:
        df_presentations (pd.DataFrame): DataFrame containing the presentations.
        topic_column (str): Column name in df_presentations that contains the topics to embed.
        cde_embeddings_model (SentenceTransformer): The CDE SentenceTransformer model to use for embedding.
    Returns:
        pd.DataFrame: DataFrame containing the embeddings of the presentation topics.
        pd.DataFrame: DataFrame containing the similarity matrix of the presentation embeddings.
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


    return df_presentation_embeddings

def calculate_similarity_matrix(df_embeddings, df_presentations, embedding_model):
    """
    Calculate the similarity matrix from the embeddings DataFrame.
    Args:
        df_embeddings (pd.DataFrame): DataFrame containing the embeddings.
        df_presentations (pd.DataFrame): DataFrame containing the presentations.
        embedding_model (SentenceTransformer): The SentenceTransformer model to use for calculating similarities.
    Returns:
        pd.DataFrame: DataFrame containing the similarity matrix.
    """
    presentation_similarities = embedding_model.similarity(df_embeddings.values, df_embeddings.values)
    # Convert similarity matrix to a dataframe (will use dataframe as it is easier to drop index values)
    df_presentation_similarities = pd.DataFrame(presentation_similarities.cpu(), index = df_presentations.index, columns=df_presentations.index)
    return df_presentation_similarities

# TODO: It may be good to return the number of presentations removed. Also, maybe the similarity and the presentations removed.
def remove_duplicates(df_presentations, df_similarities, df_embeddings, threshold=0.95):
    """
    Remove near-duplicate rows based on a similarity threshold.
    Args:
        df_presentations (pd.DataFrame): DataFrame containing presentation data.
        df_similarities (pd.DataFrame): DataFrame containing similarity scores.
        threshold (float): Similarity threshold for considering items as near duplicates.
    Returns:
        pd.DataFrame: Presentations DataFrame with near-duplicate rows removed.
        pd.DataFrame: Similarities DataFrame with near-duplicate rows and columns removed.
        pd.DataFrame: Embeddings DataFrame with near-duplicate rows removed.
    """
    # Get the indices from the dataframe (important for referencing)
    presentation_indices = df_similarities.index.tolist()
    index_map = {i: idx for i, idx in enumerate(presentation_indices)} # Map position to actual index

    # Create a set to store the indices we want to REMOVE
    indices_to_remove = set()

    # Iterate through the upper triangle of the similarity matrix to avoid redundant checks and self-comparison
    # Use iloc for positional indexing which is often faster for loops
    similarity_matrix_np = df_similarities.values # Get numpy array for faster access
    num_items = similarity_matrix_np.shape[0]

    for i in range(num_items):
        for j in range(i + 1, num_items): # Start j from i+1 to get upper triangle
            similarity_score = similarity_matrix_np[i, j]

            if similarity_score >= threshold:
                # Found a near-duplicate pair
                index_i = index_map[i] # Get the actual index value
                index_j = index_map[j] # Get the actual index value

                # Identify the index to remove (the one with the lower value)
                index_to_drop = min(index_i, index_j)
                index_to_keep = max(index_i, index_j)

                # Add the lower index to the set for removal
                indices_to_remove.add(index_to_drop)
                # Optional: print information about the identified pair
                # print(f"Near duplicate found: Index {index_i} and Index {index_j} (Similarity: {similarity_score:.4f}). Keeping {index_to_keep}, removing {index_to_drop}.")


    # Convert the set of indices to remove into a list
    indices_to_remove_list = sorted(list(indices_to_remove)) # Sorting is optional but good practice

    print(f"\nFound {len(indices_to_remove_list)} near-duplicate presentations to remove (keeping highest index).")
    print(f"Indices to remove: {indices_to_remove_list}")

    # --- Perform the removal ---

    # Keep only the rows whose indices are NOT in the removal list
    df_presentations = df_presentations.drop(index=indices_to_remove_list)
    df_embeddings = df_embeddings.drop(index=indices_to_remove_list)

    # For the square similarity matrix, remove both rows and columns
    df_similarities = df_similarities.drop(index=indices_to_remove_list, columns=indices_to_remove_list)

    # --- Verification ---
    print(f"\nFinal number of oral presentations: {len(df_presentations)}")
    print(f"Final shape of similarities matrix: {df_similarities.shape}")
    print(f"Final shape of embeddings matrix: {df_embeddings.shape}")

    # Verify indices still match
    if not df_presentations.index.equals(df_embeddings.index):
        raise ValueError("Warning: Indices of df_presentations and df_presentation_embeddings do not match!!")
    if not df_presentations.index.equals(df_similarities.index):
        raise ValueError("Warning: Row indices of df_presentations and df_presentation_similarities do not match!")
    if not df_similarities.index.equals(df_similarities.columns):
        raise ValueError("Warning: Row and Column indices of df_presentation_similarities do not match!")


    # Reset the index of the DataFrames to ensure they are clean and sequential
    df_presentations = df_presentations.reset_index()
    df_similarities.reset_index(drop=True, inplace=True)
    df_embeddings.reset_index(drop=True, inplace=True)
    return df_presentations, df_similarities, df_embeddings

def create_sessions(df_presentations, df_presentation_similarities, df_presentation_embeddings, 
                   max_sessions=100, min_session_size=8, tree_merge_stop=0.95, cluster_column_name="Session"):
    """
    Create sessions from the presentations based on their embeddings and similarities.

    Args:
        df_presentations (pd.DataFrame): DataFrame containing the presentations.
        df_presentation_similarities (pd.DataFrame): DataFrame containing the similarity matrix of the presentation embeddings.
        df_presentation_embeddings (pd.DataFrame): DataFrame containing the embeddings of the presentation topics.
        max_sessions (int): Maximum number of sessions to create.
        min_session_size (int): Minimum number of presentations in a session.
        tree_merge_stop (float): The fraction of the tree to stop clustering at.
        cluster_column_name (str): The name of the column to store cluster labels.

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
    
    # Convert to numpy arrays for faster operations
    similarity_matrix = df_presentation_similarities.values
    embeddings_array = df_presentation_embeddings.values
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(
        y=embeddings_array,
        method='average',
        metric='cosine',
    )
    
    # Pre-allocate arrays for better performance
    n_nodes = linkage_matrix.shape[0]
    n_pres = similarity_matrix.shape[0]
    
    final_clusters = []
    unassigned_count = linkage_matrix[:, 3].copy()
    unassigned_leaves = [[] for _ in range(n_nodes)]
    
    # Pre-calculate the stopping point
    merge_stop_index = int(n_nodes * tree_merge_stop)
    
    # Main clustering loop
    for i in range(n_nodes):
        left_child, right_child = linkage_matrix[i, 0:2].astype(int)
        
        # Process left child
        if left_child >= n_pres:
            left_idx = left_child - n_pres
            left_size = unassigned_count[left_idx]
            unassigned_leaves[i].extend(unassigned_leaves[left_idx])
        else:
            left_size = 1
            unassigned_leaves[i].append(left_child)
        
        # Process right child
        if right_child >= n_pres:
            right_idx = right_child - n_pres
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
            
            # Create final cluster
            final_clusters.append(unassigned_leaves[i][:])  # Make a copy
            unassigned_count[i] = 0
            unassigned_leaves[i] = []
    
    # Assign remaining unassigned leaves
    if n_nodes > 0 and unassigned_leaves[-1]:
        final_clusters = _assign_remaining_items(unassigned_leaves[-1], final_clusters, similarity_matrix)
    
    # Create outputs
    df_result, df_sessions, labels, metadata = _create_output_structures(
        final_clusters, df_presentations, cluster_column_name, n_pres
    )
 
    return df_result, df_sessions, labels, metadata

def _assign_remaining_items(unassigned_items, final_clusters, similarity_matrix):
    """Efficiently assign remaining items to best matching clusters."""
    if not final_clusters or not unassigned_items:
        return final_clusters
    
    # Pre-filter empty clusters
    non_empty_clusters = [(i, cluster) for i, cluster in enumerate(final_clusters) if cluster]
    
    if not non_empty_clusters:
        return final_clusters
    
    for leaf in unassigned_items:
        best_cluster_idx = -1
        best_similarity = -1
        
        for cluster_idx, cluster in non_empty_clusters:
            # Vectorized similarity calculation
            similarities = similarity_matrix[leaf, cluster]
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_cluster_idx = cluster_idx
        
        if best_cluster_idx >= 0:
            final_clusters[best_cluster_idx].append(leaf)
    
    return final_clusters

def _create_output_structures(final_clusters, df_presentations, cluster_column_name, n_pres):
    """Create DataFrame with cluster assignments, labels, and metadata."""
    # Initialize labels array
    labels = [-1] * n_pres
    
    # Create cluster DataFrame data
    cluster_data = []
    total_assigned = 0
    
    for cluster_id, cluster_indices in enumerate(final_clusters):
        if cluster_indices:  # Only include non-empty clusters
            # Sort indices for consistent output
            sorted_indices = sorted(cluster_indices)
            cluster_data.append({
                'cluster_id': cluster_id,
                'presentation_indices': sorted_indices,
                'cluster_size': len(sorted_indices)
            })
            
            # Assign labels
            for item_index in cluster_indices:
                labels[item_index] = cluster_id
            
            total_assigned += len(cluster_indices)
    
    # Create cluster DataFrame
    df_sessions = pd.DataFrame(cluster_data)
    
    # Create modified presentations DataFrame
    df_result = df_presentations.copy()
    df_result[cluster_column_name] = labels
    
    # Create metadata
    metadata = {
        'n_clusters': len(df_sessions),
        'n_assigned_items': total_assigned,
        'n_unassigned_items': labels.count(-1),
        'cluster_sizes': df_sessions['cluster_size'].tolist() if not df_sessions.empty else [],
        'total_presentations': len(df_result),
        'clustering_efficiency': total_assigned / len(df_result) if len(df_result) > 0 else 0
    }
    
    return df_result, df_sessions, labels, metadata

def calculate_avg_similarity(df_sessions, similarity_matrix):
    """
    Calculate average intra-cluster similarity for each cluster.
    
    Args:
        df_sessions (pd.DataFrame): DataFrame with cluster assignments
        similarity_matrix (np.ndarray): Similarity matrix of presentations
        
    Returns:
        list: Average similarity scores for each cluster
    """
    avg_similarities = []
    
    for _, row in df_sessions.iterrows():
        cluster_indices = row['presentation_indices']
        
        if len(cluster_indices) < 2:
            # Single item clusters have no internal similarity
            avg_similarities.append(np.nan)
            continue
        
        # Get all pairwise similarities within the cluster
        cluster_similarities = []
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx_i = cluster_indices[i]
                idx_j = cluster_indices[j]
                cluster_similarities.append(similarity_matrix[idx_i, idx_j])
        
        # Calculate average similarity
        avg_similarity = np.mean(cluster_similarities)
        avg_similarities.append(avg_similarity)
    
    return avg_similarities

def calculate_silhouette_scores(df_sessions, embeddings_array, labels):
    """
    Calculate silhouette scores for each cluster.
    
    Args:
        df_sessions (pd.DataFrame): DataFrame with cluster assignments
        embeddings_array (np.ndarray): Array of embeddings
        labels (list): List of cluster labels for each presentation
        
    Returns:
        list: Silhouette scores for each cluster
    """
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.metrics.pairwise import cosine_distances
    
    # Convert labels to numpy array and handle unassigned items
    labels_array = np.array(labels)
    
    # Only calculate silhouette for assigned items (exclude -1 labels)
    assigned_mask = labels_array != -1
    
    if np.sum(assigned_mask) < 2:
        # Need at least 2 assigned items
        return [np.nan] * len(df_sessions)
    
    assigned_embeddings = embeddings_array[assigned_mask]
    assigned_labels = labels_array[assigned_mask]
    
    # Check if we have at least 2 different clusters
    unique_labels = np.unique(assigned_labels)
    if len(unique_labels) < 2:
        return [np.nan] * len(df_sessions)
    
    # Calculate distances using cosine distance
    distances = cosine_distances(assigned_embeddings)
    
    # Calculate silhouette scores for each sample
    sample_scores = silhouette_samples(distances, assigned_labels, metric='precomputed')
    
    # Calculate average silhouette score for each cluster
    cluster_silhouette_scores = []
    
    for _, row in df_sessions.iterrows():
        cluster_id = row['cluster_id']
        
        # Find samples belonging to this cluster in the assigned data
        cluster_mask = assigned_labels == cluster_id
        
        if np.sum(cluster_mask) > 0:
            cluster_score = np.mean(sample_scores[cluster_mask])
            cluster_silhouette_scores.append(cluster_score)
        else:
            cluster_silhouette_scores.append(np.nan)
    
    return cluster_silhouette_scores

def calculate_document_similarities(similarity_matrix, labels):
    """
    
    Calculate average similarity of each document to others in its cluster
    This version have been vectorized for better performance with large datasets
    Args:
        similarity_matrix (np.ndarray): Similarity matrix of shape (n_samples, n_samples)
        labels (array-like): Cluster labels for each document
    
    Returns:
        np.ndarray: Average similarity of each document to its cluster
    """
        # Ensure labels is a numpy array for proper vectorized operations
    labels = np.array(labels)
    
    # Input validation
    n_samples = len(labels)
    if similarity_matrix.shape[0] != n_samples or similarity_matrix.shape[1] != n_samples:
        raise ValueError(f"Similarity matrix shape {similarity_matrix.shape} doesn't match labels length {n_samples}")
    document_similarities = np.zeros(n_samples)
    
    # Process each unique cluster
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude unassigned
    
    for cluster_label in unique_labels:
        cluster_mask = labels == cluster_label
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 1:  # Only process multi-item clusters
            # Extract submatrix for this cluster
            cluster_sim_matrix = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
            
            # Calculate mean similarity for each item (excluding diagonal)
            cluster_means = (cluster_sim_matrix.sum(axis=1) - np.diag(cluster_sim_matrix)) / (len(cluster_indices) - 1)
            
            # Assign back to main array
            document_similarities[cluster_indices] = cluster_means
    
    return document_similarities

# Default prompts as module-level constants
DEFAULT_SESSION_PROMPT = """I am organizing oral research presentation sessions for the American Society of Biological and Agricultural Engineers Annual International Meeting. Please provide 3 options for the name/title of a session. Also provide 5 keywords describing the session. The name and keywords should highlight the commonality among all presentations. The target audience for titles and keywords is engineering designers and researchers. The title should be descriptive of the content and be interesting and engaging.

Please respond in JSON format:
{{"title1": "Session Title", "title2": "Session Title", "title3": "Session Title", "keywords": "keyword1, keyword2, keyword3, keyword4, keyword5"}}

The titles and abstracts for presentations assigned to this session are:
{presentations}"""

DEFAULT_GEMINI_SESSION_PROMPT = """I am organizing oral research presentation sessions for the American Society of Biological and Agricultural Engineers Annual International Meeting. Please provide 3 options for the name/title of a session. Also provide 5 keywords describing the session. The name and keywords should highlight the commonality among all presentations. The target audience for titles and keywords is engineering designers and researchers. The title should be descriptive of the content and be interesting and engaging.

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
        session_num = row["cluster_id"]
        presentation_indices = row["presentation_indices"]
        
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
                    session_value["Ollama Title 1"] = answer_dict.get("title1", f"No Title Generated")
                    session_value["Ollama Title 2"] = answer_dict.get("title2", f"No Title Generated")
                    session_value["Ollama Title 3"] = answer_dict.get("title3", f"No Title Generated")
                    session_value["Ollama Keywords"] = answer_dict.get("keywords", "No Keywords Generated")
                    
                    print(f"  ✓ Generated titles for session {session_key}")
                    
                except json.JSONDecodeError as e:
                    print(f"  ✗ JSON parsing failed for session {session_key}: {e}")
                    print(f"  Raw response: {text_answer[:1000]}...")
                    
                    # Set fallback values
                    session_value["Ollama Title 1"] = f"No Title Generated"
                    session_value["Ollama Title 2"] = f"No Title Generated"
                    session_value["Ollama Title 3"] = f"No Title Generated"
                    session_value["Ollama Keywords"] = "No Keywords Generated"
                    
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"  ✗ Error processing session {session_key}: {e}")
            # Set fallback values
            session_value["Ollama Title 1"] = f"No Title Generated"
            session_value["Ollama Title 2"] = f"No Title Generated"
            session_value["Ollama Title 3"] = f"No Title Generated"
            session_value["Ollama Keywords"] = "No Keywords Generated"
    
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
        session_id = row["cluster_id"]
        if session_id in sessions_dict:
            title_1_list.append(sessions_dict[session_id].get("Ollama Title 1", "No Title Generated"))
            title_2_list.append(sessions_dict[session_id].get("Ollama Title 2", "No Title Generated"))
            title_3_list.append(sessions_dict[session_id].get("Ollama Title 3", "No Title Generated"))
            keywords_list.append(sessions_dict[session_id].get("Ollama Keywords", "No Keywords Generated"))
        else:
            title_1_list.append("No Title Generated")
            title_2_list.append("No Title Generated")
            title_3_list.append("No Title Generated")
            keywords_list.append("No Keywords Generated")
    
    # Add new columns to DataFrame
    df_sessions_with_titles["Ollama Title 1"] = title_1_list
    df_sessions_with_titles["Ollama Title 2"] = title_2_list
    df_sessions_with_titles["Ollama Title 3"] = title_3_list
    df_sessions_with_titles["Ollama Keywords"] = keywords_list
    
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
):
    """
    Generate session titles and keywords using Google's Gemini API.

    Args:
        df_sessions (pd.DataFrame): DataFrame with session information from create_sessions()
        df_presentations (pd.DataFrame): DataFrame with presentation data
        topic_column (str): Column name containing the combined title and abstract text
        prompt_template (str): Custom prompt template with {presentations} placeholder

    Returns:
        pd.DataFrame: df_sessions with added columns for generated titles and keywords
    """
    # Load environment variables
    load_dotenv(".env")

    # Use default prompt if none provided
    if prompt_template is None:
        prompt_template = DEFAULT_GEMINI_SESSION_PROMPT

    # Check for API key
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. Please set it in your .env file."
        )

    # Create sessions dictionary from df_sessions
    sessions_dict = {}
    for _, row in df_sessions.iterrows():
        session_num = row["cluster_id"]
        presentation_indices = row["presentation_indices"]

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
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

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
        session_id = row["cluster_id"]
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
        session_num = row["cluster_id"]
        presentation_indices = row["presentation_indices"]

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
        session_id = row["cluster_id"]
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
):
    """
    Generate session titles and keywords using various AI models.

    Args:
        df_sessions: DataFrame with session information
        df_presentations: DataFrame with presentation data
        topic_column: Column name containing presentation text
        model_name: Model to use ('gemini-2.0-flash', 'llama-3.2-local', 'ollama:model_name')
        prompt_template: Custom prompt template with {presentations} placeholder
    """
    if model_name == "gemini-2.0-flash":
        return generate_session_titles_and_keywords_gemini(
            df_sessions, df_presentations, topic_column, prompt_template
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
    
# You can also add error handling and validation
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
        committee_embeddings = committee_embeddings.values

    # Pre-normalize committee embeddings once
    committee_norms = committee_embeddings / np.linalg.norm(
        committee_embeddings, axis=1, keepdims=True
    )

    results = []

    for _, cluster_row in df_sessions.iterrows():
        session_id = cluster_row["cluster_id"]
        presentation_indices = cluster_row["presentation_indices"]

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
        committee_pivot, left_on="cluster_id", right_index=True, how="left"
    )


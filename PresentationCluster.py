import pandas as pd
from sentence_transformers import SentenceTransformer
import time
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import pickle


df = pd.read_excel("1.29.25 Abstracts.xlsx")
df = df.rename(columns={'Session':'Original Session',
                        'Submission Name':'Title',
                        'Abstract-Character max 4000-Abstracts will only be used to evaluate quality of talk and topic. They will not be published or able to be edited later.':'Abstract',
                        'Submission ID - 7 digits':'Abstract ID',
                        'Technical Community':'Original Technical Community',
                        'Profile: First Name':'First Name',
                        'Profile: Last Name':'Last Name',
                        })

# Drop any presentations that are missing an abstrct or title
df.dropna(subset=['Title','Abstract'], inplace=True)
# Before beginning, make sure all presentations have a session assigned. 
# Unassigned presentations are given the session name, unassigned.
df['Original Session'] = df['Original Session'].fillna('UNASSIGNED')
# Combine Titles and Abstracts with a semicolon in between. 
# This should be the same as + but .agg() handles empty fields or fields that have non-text entries.
df['Title and Abstract'] = df[['Title', 'Abstract',]].agg(': '.join, axis=1)
# print(f"Title and Abstract: \n{df['Title and Abstract'].head()}")
# Select the oral presentations only
df_presentations = df[~df['Original Session'].str.contains('POSTER', case=False, na=False)].copy()

# Load the CDE embeddings model
embedding_model = SentenceTransformer('jxm/cde-small-v1', trust_remote_code=True)
topic_column = 'Title and Abstract'

def embed_presentations(df_presentations, topic_column, embedding_model):

    return cde_embed_presentations(df_presentations, topic_column, embedding_model)

def cde_embed_presentations(df_presentations, topic_column, cde_embeddings_model):
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
        # Put some strings here that are representative of your corpus, for example by calling random.sample(corpus, k=minicorpus_size)
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


    start_time = time.time()
    cde_dataset_embeddings = cde_embeddings_model.encode(
        minicorpus_docs,
        prompt_name="document",
        convert_to_tensor=True
    )
    end_time = time.time()
    cde_dataset_elapsed_time = end_time - start_time
    # print(f"Titles Embedded: {len(cde_dataset_embeddings)}, Length of Vectors: {len(cde_dataset_embeddings[0])}")
    # print(f"Dataset Encoding time: {cde_dataset_elapsed_time:.4f} seconds")

    # Now embed the titles and abstracts using the CDE embeddings model
    start_time = time.time()
    presentation_embeddings = cde_embeddings_model.encode(
        presentation_topics,
        prompt_name="document",
        dataset_embeddings=cde_dataset_embeddings,
        convert_to_tensor=True,
    )
    end_time = time.time()
    cde_doc_elapsed_time = end_time - start_time
    # print(f"Titles Embedded: {len(presentation_embeddings)}, Length of Vectors: {len(presentation_embeddings[0])}")
    # print(f"Document Encoding time: {cde_doc_elapsed_time:.4f} seconds")
    cde_elapsed_time = cde_doc_elapsed_time+cde_dataset_elapsed_time
    # print(f"Total Encoding time: {(cde_elapsed_time):.4f} seconds")
    presentation_similarities = cde_embeddings_model.similarity(presentation_embeddings, presentation_embeddings)
    # Convert similarity matrix to a dataframe (will use dataframe as it is easier to drop index values)
    df_presentation_similarities = pd.DataFrame(presentation_similarities.cpu(), index = df_presentations.index, columns=df_presentations.index)
    df_presentation_embeddings = pd.DataFrame(presentation_embeddings.cpu(), index = df_presentations.index)


    return df_presentation_similarities, df_presentation_embeddings

df_presentation_similarities, df_presentation_embeddings = embed_presentations(df_presentations, topic_column, embedding_model)

# Use similarities to remove duplicates or near-duplicates
def remove_duplicates(df_presentations, df_similarities, df_embeddings, threshold=0.95):
    """
    Remove near-duplicate rows based on a similarity threshold.
    Args:
        df_presentations (pd.DataFrame): DataFrame containing presentation data.
        df_similarities (pd.DataFrame): DataFrame containing similarity scores.
        threshold (float): Similarity threshold for considering items as near duplicates.
    Returns:
        pd.DataFrame: Presentations DataFrame with near-duplicate rows removed.


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

            if similarity_score >= similarity_threshold:
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

    # print(f"\nFound {len(indices_to_remove_list)} near-duplicate presentations to remove (keeping highest index).")
    # print(f"Indices to remove: {indices_to_remove_list}")

    # --- Perform the removal ---

    # Keep only the rows whose indices are NOT in the removal list
    df_presentations = df_presentations.drop(index=indices_to_remove_list)
    df_embeddings = df_embeddings.drop(index=indices_to_remove_list)

    # For the square similarity matrix, remove both rows and columns
    df_similarities = df_similarities.drop(index=indices_to_remove_list, columns=indices_to_remove_list)

    # --- Verification ---
    # print(f"\nFinal number of oral presentations: {len(df_presentations)}")
    # print(f"Final shape of similarities matrix: {df_similarities.shape}")
    # print(f"Final shape of embeddings matrix: {df_presentation_embeddings.shape}")

    # Verify indices still match
    if not df_presentations.index.equals(df_embeddings.index):
        print("Warning: Indices of df_presentations and df_presentation_embeddings do not match!")
    if not df_presentations.index.equals(df_similarities.index):
        print("Warning: Row indices of df_presentations and df_presentation_similarities do not match!")
    if not df_similarities.index.equals(df_similarities.columns):
        print("Warning: Row and Column indices of df_presentation_similarities do not match!")


    # Reset the index of the DataFrames to ensure they are clean and sequential
    df_presentations = df_presentations.reset_index()
    df_similarities.reset_index(drop=True, inplace=True)
    df_embeddings.reset_index(drop=True, inplace=True)
    return df_presentations, df_similarities, df_embeddings


# Define the similarity threshold for considering items as near duplicates
# Difference < 0.01 means Similarity >= 0.99
similarity_threshold = 0.99    
remove_duplicates(df_presentations, df_presentation_similarities, df_presentation_embeddings, similarity_threshold)

# # Convert the similarities DataFrame to a NumPy array (this will drop/reset the index and columns)
# oral_similarities_np = df_presentation_similarities.values
# # Convert the NumPy array to a PyTorch tensor
# oral_similarities_tensor = torch.tensor(oral_similarities_np, dtype=torch.float32)
# df_similarity = pd.DataFrame(oral_similarities_tensor.cpu(), index = df_presentations.index, columns=df_presentations.index)

def calculate_cluster_similarities(similarity_matrix, labels):
    """
    Calculate average similarity of each document with others in its cluster
    
    Returns:
    - document_similarities: Average similarity of each document with its cluster
    - cluster_avg_similarities: Average similarity for each cluster
    """
    n_samples = len(labels)
    document_similarities = np.zeros(n_samples)
    cluster_avg_similarities = {}
    
    for i in range(n_samples):
        # Get indices of other documents in the same cluster
        cluster_idx = np.where(labels == labels[i])[0]
        cluster_idx = cluster_idx[cluster_idx != i]  # Exclude self
        
        if len(cluster_idx) > 0:  # If there are other documents in the cluster
            # Calculate average similarity with other documents in cluster
            document_similarities[i] = np.mean(similarity_matrix[i, cluster_idx])
        
    # Calculate average similarity for each cluster
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        cluster_mask = labels == cluster
        cluster_docs = np.where(cluster_mask)[0]
        
        if len(cluster_docs) > 1:
            cluster_similarities = []
            for doc in cluster_docs:
                other_docs = cluster_docs[cluster_docs != doc]
                avg_sim = np.mean(similarity_matrix[doc, other_docs])
                cluster_similarities.append(avg_sim)
            cluster_avg_similarities[cluster] = np.mean(cluster_similarities)
        else:
            cluster_avg_similarities[cluster] = 0.0
            
    return document_similarities, cluster_avg_similarities


def create_session_index_lists(df_presentations, session_column='Original Session'):
    """
    Creates a list of lists, where each inner list contains the indices of presentations
    assigned to a particular session.

    Args:
        df_presentations (pd.DataFrame): The DataFrame containing presentation data.
        session_column (str): The name of the column containing session names.

    Returns:
        list: A list of lists, where each inner list contains indices for a session.
        dict: A dictionary mapping session names to their corresponding indices.
    """
    # Reset index to ensure consistent indexing (similarity tensor is re-indexed)
    df_presentations.reset_index(drop=True, inplace=True)  
    session_indices = {}
    for index, session in df_presentations[session_column].items():
        if session not in session_indices:
            session_indices[session] = []
        session_indices[session].append(index)

    return list(session_indices.values()), session_indices

def analyze_sessions(sessions_dict, similarities):
    """
    Calculates statistics for sessions using a dictionary and a NumPy similarity matrix.

    Args:
        sessions_dict (dict): A dictionary where keys are session names 
                              and values are lists of indices of items in that session.
        similarities (np.ndarray): The cosine similarity matrix as a NumPy array.

    Returns:
        dict: A dictionary containing the calculated statistics:
              - 'avg_similarity': Overall average similarity across sessions with > 1 item.
              - 'min_session_similarity': Minimum average similarity found in any session with > 1 item.
              - 'num_sessions': Total number of sessions.
              - 'clusters_gt1': Number of sessions with more than one item.
              - 'session_sizes': A dictionary mapping session names to their sizes.
              - 'similarity_values': A dictionary mapping session names to their average internal similarity.
    """
    # Initialize dictionaries to store results for each session
    session_similarities = {}
    session_sizes = {}
    clusters_gt1 = 0 # Counter for sessions with more than one item

    # Iterate through each session in the input dictionary
    for session_name, session_indices in sessions_dict.items():
        # Store the size (number of items) of the current session
        session_sizes[session_name] = len(session_indices)
        
        # Calculate similarity only for sessions with more than one item
        if len(session_indices) > 1:
            clusters_gt1 += 1 # Increment the counter for sessions > size 1
            
            # Use numpy advanced indexing (np.ix_) to extract the submatrix 
            # corresponding to the current session's indices
            session_similarity_matrix = similarities[np.ix_(session_indices, session_indices)]
            
            # Calculate the sum of the upper triangle of the submatrix (excluding the diagonal)
            # This sums the similarities between unique pairs within the session
            similarity_sum = np.sum(np.triu(session_similarity_matrix, k=1))
            
            # Calculate the number of unique pairs in the session
            num_pairs = len(session_indices) * (len(session_indices) - 1) / 2
            
            # Calculate the average similarity for the session
            # Handle the case where num_pairs might be zero (although guarded by len > 1 check)
            avg_session_similarity = similarity_sum / num_pairs if num_pairs > 0 else 0.0
            
            # Store the calculated average similarity for the session
            session_similarities[session_name] = avg_session_similarity
            
        # For sessions with only one item, similarity is not meaningful (set to 0)
        elif len(session_indices) == 1:
            session_similarities[session_name] = 0.0 
        # For empty sessions, similarity is 0
        else:
            session_similarities[session_name] = 0.0 

    # Calculate the overall average similarity and minimum similarity across sessions with > 1 item.
    # Iterate through the calculated similarities, checking the original session size.
    valid_similarities = [
        sim for session_name, sim in session_similarities.items() 
        if len(sessions_dict.get(session_name, [])) > 1 
    ] # Filter to include only similarities from sessions with more than one item.
    
    overall_avg_similarity = np.mean(valid_similarities) if valid_similarities else 0.0
    min_session_similarity = min(valid_similarities) if valid_similarities else 0.0

    # Compile the results into a dictionary
    results = {
        'avg_similarity': overall_avg_similarity,
        'min_session_similarity': min_session_similarity,
        'num_sessions': len(sessions_dict),
        'clusters_gt1': clusters_gt1,
        'session_sizes': session_sizes, # Dictionary of session_name: size
        'similarity_values': session_similarities, # Dictionary of session_name: avg_similarity
    }

    return results


# pres_session_similarity, session_similarity = calculate_cluster_similarities(oral_similarities_tensor.cpu().numpy(), np.array(df_presentations['Original Session']))
# # print(f"The average session similarity after submission is {np.mean(list(session_similarity.values())):.3f}")
# original_session_list, original_session_dict = create_session_index_lists(df_presentations)
# original_analysis_results = analyze_sessions(original_session_dict, oral_similarities_tensor.cpu().numpy())


from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
linkage_matrix = linkage(
    y=df_presentation_embeddings,
    method='average',
    metric='cosine',
    )

# Create sessions using the agglomerative clustering linkage matrix. 
# Start merging with the most similar presentations (top of the linkage matrix).
# Once the clusters are at min_cluster_size, stop automatically merging.



def extract_clusters(linkage_matrix, min_cluster_size, n_samples, similarities, tree_merge_stop=1, max_sessions=None):
    """
    Extracts clusters from a linkage matrix, ensuring no cluster is smaller than min_cluster_size.

    Args:
        linkage_matrix: The linkage matrix from scipy.cluster.hierarchy.linkage.
        min_cluster_size: The minimum number of samples in a cluster.
        n_samples: The total number of samples.
        similarities: The similarity matrix tensor.
        tree_merge_stop: The fraction of the tree to stop clustering at.
        max_sessions: Maximum number of sessions to create. If None, there is no limit.

    Returns:
        A list of lists, where each inner list represents a cluster and contains
        the indices of the samples belonging to that cluster.

    """

    final_nodes = []
    final_clusters = []
    unassigned_count = linkage_matrix[:,3].copy() # Number of unassigned clusters. All nodes start unassigned.
    unassigned_leaves = [[] for _ in range(linkage_matrix.shape[0])] # Leaves that have not yet been assigned to a final cluster.
    # Iterate through the linkage matrix (Linkage Matrix (begin -> end) goes bottom -> top on the dendrogram)
    # Process starts with the closest (most similar) nodes/clusters and works its way up the hierarchy.
    # Each row of the linkage matrix represents a merge of two clusters.
    # At each step, check the number of unassigned clusters (unassigned_count) for the two children of the current node.
    # If the number of unassigned clusters for either child is greater than min_cluster_size,
    # then we can create a final cluster (session).
    # If the number of unassigned clusters for both children is less than min_cluster_size,
    # leave them unassigned. Record the unassigned value as the amount remaining unassigned at that node.
    # Continue to the next row of the linkage matrix.

    for i in range(linkage_matrix.shape[0]):
        node_id = n_samples + i
        left_child, right_child = linkage_matrix[i, 0:2].astype(int)
        # Check children to update unassigned_count for this node
        if left_child >= n_samples:
            left_size = unassigned_count[left_child - n_samples]
            unassigned_leaves[i].extend(unassigned_leaves[left_child - n_samples])
        else:
            left_size = 1
            unassigned_leaves[i].append(left_child)
        if right_child >= n_samples:
            right_size = unassigned_count[right_child - n_samples]
            unassigned_leaves[i].extend(unassigned_leaves[right_child - n_samples])
        else:
            right_size = 1
            unassigned_leaves[i].append(right_child)
        unassigned_count[i] = left_size + right_size
        
        # Check if we have a final cluster, 
        # Also, stop making clusters once we are past a certain point in the linkage matrix.
        # Or, stop making clusters once we reach the maximum sessions.
        if unassigned_count[i] > min_cluster_size:
            # Check tree merge stop and max sessions conditions separately
            if i < linkage_matrix.shape[0] * tree_merge_stop:
                # Tree merge stop condition met; Check to make sure we are not at session cap.
                if max_sessions is None or len(final_clusters) < max_sessions:
                    # Under max sessions limit (or no limit)
                    create_cluster = True
                else:
                    create_cluster = False

            if create_cluster:
                # A final cluster has been found. All the presentations (unassigned leaves) in this node are assigned to it.
                final_nodes.append(node_id)
                final_clusters.append(unassigned_leaves[i])
                # Remove all presentations in this node from the unassigned_count.
                # When checked as a child of another node, it will report 0.
                unassigned_count[i] = 0
                unassigned_leaves[i] = []
    # Keep track of the presentations that were assigned in this final step.
    # They are the outliers and should probably be looked at closer.
    forced_merge = unassigned_leaves    
    # Try to merge any unassigned leaves into other clusters.
    # Since we stop merging at a certain point in the linkage matrix, we may have some unassigned leaves left over.
    # These are the leaves that were not assigned to any final cluster.
    # This does not check if the final clusters are too large.
    if unassigned_leaves[-1]:
        for leaf in unassigned_leaves[-1]:
            best_cluster_index = -1
            best_similarity = -1
            for i, cluster in enumerate(final_clusters):
                avg_similarity = similarities[leaf, cluster].mean().item()
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_cluster_index = i
            if best_cluster_index > -1:
                final_clusters[best_cluster_index].append(leaf)
        unassigned_leaves[-1] = []

    return final_clusters, linkage_matrix, forced_merge, final_nodes

def process_clusters(clusters):
    """
    Processes a list of cluster indices to create a cluster dictionary and labels list.

    Args:
        clusters: A list of lists, where each inner list contains the indices
                        of items belonging to a cluster.

    Returns:
        A tuple containing:
            - cluster_dict: A dictionary where keys are cluster IDs (integers) and
                          values are lists of item indices belonging to that cluster.
            - labels: A list (or Pandas Series) where each element at index i
                      represents the cluster assignment of item i.
    """
    # Find the maximum index value across all clusters
    max_index = 0
    for cluster in clusters:
        if cluster and max(cluster) > max_index:
            max_index = max(cluster)
    
    # Initialize labels list with None values of appropriate size
    labels = [None] * (max_index + 1)  # +1 because indices are 0-based

    cluster_dict = {}
    for cluster_id, cluster_indices in enumerate(clusters):
        cluster_dict[cluster_id] = cluster_indices
        for item_index in cluster_indices:
            labels[item_index] = cluster_id  # Assign cluster ID to the item index

    return cluster_dict, labels


min_cluster_size = 8
tree_merge_stop = 1
max_num_sessions = 100
session_indices, link_mat_final, outliers_merged, fnode_final = extract_clusters(linkage_matrix, 
                                                                                min_cluster_size, 
                                                                                n_samples=len(oral_similarities_tensor), 
                                                                                similarities=oral_similarities_tensor, 
                                                                                tree_merge_stop=tree_merge_stop, 
                                                                                max_sessions=max_num_sessions
                                                                                )

cluster_dict, labels = process_clusters(session_indices)
analysis_results_after_cluster = analyze_sessions(cluster_dict, oral_similarities_tensor.cpu().numpy())

cluster_dict, labels = process_clusters(session_indices)
# Replace None values in the labels list with a default integer value (e.g., -1)
labels = [label if label is not None else -1 for label in labels]

# Assign the labels to the DataFrame and explicitly set the data type to int
df_presentations['Clustering Session'] = pd.Series(labels, dtype='int')

cluster_session_list, cluster_session_dict = create_session_index_lists(df_presentations, session_column='Clustering Session')

# Create a dict for the cluster indices
cluster_dict = df_presentations.groupby('Clustering Session').indices
# Convert keys to integers because Pandas creates them as different types.
cluster_dict = {int(k): np.array(v, dtype=np.int64).tolist() for k, v in cluster_dict.items()}
sessions_dict = {}
for session_num, presentation_indices in cluster_dict.items():
    sessions_dict[session_num] = {"Indices": presentation_indices, "Presentations": []}
    for index in presentation_indices:
        try:
            title_abstract = df_presentations.loc[index, "Title and Abstract"]
            sessions_dict[session_num]["Presentations"].append(title_abstract)
        except KeyError:
            print(f"Warning: Index {index} not found in df_presentations for session {session_num}. Skipping.")
            continue # Skip the index if it's not present in the dataframe

def calculate_session_similarity(presentations, similarity_matrix, session_column='Original Session'):
    """
    Calculates the average similarity between presentations in different sessions.

    Args:
      presentations: A pandas DataFrame with presentation information.
      similarity_matrix: A NumPy array representing the pairwise similarity between presentations.
      session_column: The column name in the DataFrame that specifies the session.

    Returns:
      A tuple containing:
        - A NumPy array representing the session-level similarity matrix.
        - A NumPy array of unique session names in the order they appear in the similarity matrix.
    """
    # Create a new presentation dataframe that has constant indexing.
    presentations = presentations.reset_index()
    sessions = presentations[session_column].unique()
    num_sessions = len(sessions)
    session_similarity = np.zeros((num_sessions, num_sessions))

    for i, session1 in enumerate(sessions):
        for j, session2 in enumerate(sessions):
            pres_idx1 = presentations[presentations[session_column] == session1].index
            pres_idx2 = presentations[presentations[session_column] == session2].index
            session_similarity[i, j] = np.mean(similarity_matrix[np.ix_(pres_idx1, pres_idx2)])

    return session_similarity, sessions

# Drop the unnecessary "index" column if it exists
if 'index' in df_presentations.columns:
    df_presentations = df_presentations.drop(columns=['index'])

pres_clustering_session_similarity, clustering_session_similarity = calculate_cluster_similarities(df_similarity.to_numpy(), np.array(df_presentations['Clustering Session']))

df_presentations['Presentation-Session Similarity - Clustering'] = pres_clustering_session_similarity
df_presentations['Session Similarity - Clustering'] = df_presentations['Clustering Session'].map(clustering_session_similarity)

# Calculate standard deviation for each 'Clustering Session'
df_presentations['Session Std Dev - Clustering'] = df_presentations.groupby('Clustering Session')['Presentation-Session Similarity - Clustering'].transform('std')
# Calculate deviation from the mean
df_presentations['Raw Deviation - Clustering'] = df_presentations['Presentation-Session Similarity - Clustering'] - df_presentations['Session Similarity - Clustering']
# Calculate standardized deviation
df_presentations['Standardized Deviation - Clustering'] = df_presentations['Raw Deviation - Clustering'] / df_presentations['Session Std Dev - Clustering']

# Create a session similarity matrix that contains the similarities from the clustering sessions.
clustering_session_similarity_matrix, clustering_sessions = calculate_session_similarity(df_presentations, df_similarity.to_numpy(), session_column='Clustering Session')
df_clustering_session_similarity = pd.DataFrame(clustering_session_similarity_matrix, index=clustering_sessions, columns=clustering_sessions)

cluster_session_list, cluster_session_dict = create_session_index_lists(df_presentations, session_column='Clustering Session')
cluster_analysis_results = analyze_sessions(cluster_session_dict, df_similarity.to_numpy())

# Export Data
# Embeddings and similarities
df_presentation_similarities.to_pickle('AIM2025_similarities.pkl')
df_presentation_embeddings.to_pickle('AIM2025_embeddings.pkl')

df_similarity.to_pickle('app_presentation_similarities.pkl')
df_presentation_embeddings.to_pickle('app_presentation_embeddings.pkl')

df_presentations.to_pickle('aim2025_clustered_presentations.pkl')
# For the web app
df_presentations_no_abstracts = df_presentations.drop(columns=['Title and Abstract','Abstract'])
df_presentations_no_abstracts.to_pickle('aim2025_clustered_presentations_no_abstracts.pkl')

with open('Cluster_analysis_resultsAIM25.pkl', 'wb') as f:
    pickle.dump(cluster_analysis_results, f)  

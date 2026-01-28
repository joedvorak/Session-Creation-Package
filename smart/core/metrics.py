"""
Metrics calculations for session organization.

Provides functions to evaluate:
- Session coherence (how similar presentations within a session are)
- Presentation fit (how well a presentation fits its assigned session)
- Session distinctiveness (how different a session is from others)
- Session-to-session similarity
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def calculate_session_coherence(
    embeddings: np.ndarray,
    session_indices: List[int],
) -> float:
    """
    Calculate coherence score for a session.
    
    Coherence is the mean pairwise similarity between all presentations
    in the session. Higher values indicate more thematically focused sessions.
    
    Args:
        embeddings: NxD array of all presentation embeddings
        session_indices: Indices of presentations in this session
        
    Returns:
        Coherence score (0 to 1, higher is more coherent)
    """
    if len(session_indices) < 2:
        return 1.0  # Single-item sessions are perfectly coherent
    
    # Extract session embeddings
    session_embeddings = embeddings[session_indices]
    
    # Calculate pairwise similarities
    similarities = cosine_similarity(session_embeddings, session_embeddings)
    
    # Mean of upper triangle (excluding diagonal)
    n = len(session_indices)
    upper_tri_sum = 0.0
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            upper_tri_sum += similarities[i, j]
            count += 1
    
    result = upper_tri_sum / count if count > 0 else 1.0
    # Convert numpy scalar to Python float for SQLite compatibility
    return float(result)


def calculate_presentation_fit(
    embedding: np.ndarray,
    session_embeddings: np.ndarray,
    exclude_self: bool = True,
) -> float:
    """
    Calculate how well a presentation fits its session.
    
    Fit is the mean similarity between the presentation and other
    presentations in the same session.
    
    Args:
        embedding: 1xD embedding of the presentation
        session_embeddings: NxD embeddings of all presentations in session
        exclude_self: Whether to exclude the presentation itself
        
    Returns:
        Fit score (0 to 1, higher means better fit)
    """
    if len(session_embeddings) == 0:
        return 0.0
    
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)
    
    similarities = cosine_similarity(embedding, session_embeddings)[0]
    
    if exclude_self and len(session_embeddings) > 1:
        # Find and exclude the self-similarity (should be ~1.0)
        max_idx = np.argmax(similarities)
        if similarities[max_idx] > 0.99:
            similarities = np.delete(similarities, max_idx)
    
    return float(np.mean(similarities)) if len(similarities) > 0 else 0.0


def calculate_session_distinctiveness(
    session_embeddings: np.ndarray,
    other_session_embeddings: List[np.ndarray],
) -> float:
    """
    Calculate how distinct a session is from other sessions.
    
    Distinctiveness is 1 minus the maximum similarity to any other session.
    Higher values indicate the session covers a unique topic area.
    
    Args:
        session_embeddings: NxD embeddings for this session
        other_session_embeddings: List of NxD embeddings for other sessions
        
    Returns:
        Distinctiveness score (0 to 1, higher is more distinct)
    """
    if len(other_session_embeddings) == 0:
        return 1.0
    
    # Calculate session centroid
    centroid = np.mean(session_embeddings, axis=0, keepdims=True)
    
    # Calculate similarity to other session centroids
    max_similarity = 0.0
    
    for other_embs in other_session_embeddings:
        if len(other_embs) == 0:
            continue
        other_centroid = np.mean(other_embs, axis=0, keepdims=True)
        sim = cosine_similarity(centroid, other_centroid)[0, 0]
        max_similarity = max(max_similarity, sim)
    
    # Convert to Python float for SQLite compatibility
    return float(1.0 - max_similarity)


def calculate_session_session_similarity(
    session_embeddings_list: List[np.ndarray],
    session_ids: List[str],
) -> Dict[Tuple[str, str], float]:
    """
    Calculate pairwise similarity between all sessions.
    
    Uses session centroids for comparison.
    
    Args:
        session_embeddings_list: List of NxD embeddings for each session
        session_ids: List of session IDs
        
    Returns:
        Dict mapping (session_id_1, session_id_2) to similarity score
    """
    n_sessions = len(session_embeddings_list)
    
    # Calculate centroids
    centroids = []
    for embs in session_embeddings_list:
        if len(embs) > 0:
            centroids.append(np.mean(embs, axis=0))
        else:
            centroids.append(np.zeros(session_embeddings_list[0].shape[1]))
    
    centroids = np.array(centroids)
    
    # Calculate all pairwise similarities
    similarities = cosine_similarity(centroids, centroids)
    
    result = {}
    for i in range(n_sessions):
        for j in range(i + 1, n_sessions):
            result[(session_ids[i], session_ids[j])] = float(similarities[i, j])
            result[(session_ids[j], session_ids[i])] = float(similarities[i, j])
    
    return result


def calculate_all_metrics(
    embeddings: np.ndarray,
    abstract_ids: List[str],
    session_assignments: Dict[str, str],
) -> Dict[str, Any]:
    """
    Calculate all metrics for a complete session organization.
    
    Args:
        embeddings: NxD array of all presentation embeddings
        abstract_ids: List of abstract IDs corresponding to embeddings
        session_assignments: Dict mapping abstract_id to session_id
        
    Returns:
        Dict with session_metrics, presentation_metrics, and summary
    """
    # Build index mappings
    id_to_idx = {aid: i for i, aid in enumerate(abstract_ids)}
    
    # Group presentations by session
    sessions: Dict[str, List[int]] = {}
    for abstract_id, session_id in session_assignments.items():
        if abstract_id not in id_to_idx:
            continue
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(id_to_idx[abstract_id])
    
    # Calculate session metrics
    session_metrics = {}
    session_embeddings_list = []
    session_ids = []
    
    for session_id, indices in sessions.items():
        session_embs = embeddings[indices]
        session_embeddings_list.append(session_embs)
        session_ids.append(session_id)
        
        coherence = calculate_session_coherence(embeddings, indices)
        
        session_metrics[session_id] = {
            "coherence": coherence,
            "size": len(indices),
            "presentation_ids": [abstract_ids[i] for i in indices],
        }
    
    # Calculate distinctiveness
    for i, session_id in enumerate(session_ids):
        other_embs = [e for j, e in enumerate(session_embeddings_list) if j != i]
        distinctiveness = calculate_session_distinctiveness(
            session_embeddings_list[i], other_embs
        )
        session_metrics[session_id]["distinctiveness"] = distinctiveness
    
    # Calculate session-session similarity
    session_similarity = calculate_session_session_similarity(
        session_embeddings_list, session_ids
    )
    
    # Calculate presentation metrics and session std_dev
    presentation_metrics = {}
    session_fit_scores = {sid: [] for sid in sessions.keys()}  # Collect fit scores per session
    
    for abstract_id, session_id in session_assignments.items():
        if abstract_id not in id_to_idx:
            continue
        
        idx = id_to_idx[abstract_id]
        session_indices = sessions[session_id]
        session_embs = embeddings[session_indices]
        
        fit = calculate_presentation_fit(embeddings[idx], session_embs)
        session_fit_scores[session_id].append(fit)
        
        presentation_metrics[abstract_id] = {
            "session_id": session_id,
            "fit": fit,
        }
    
    # Calculate session std_dev from fit scores
    for session_id, fit_scores in session_fit_scores.items():
        if len(fit_scores) > 1:
            session_metrics[session_id]["std_dev"] = float(np.std(fit_scores, ddof=1))
        else:
            session_metrics[session_id]["std_dev"] = 0.0
    
    # Summary statistics
    coherence_values = [m["coherence"] for m in session_metrics.values()]
    fit_values = [m["fit"] for m in presentation_metrics.values()]
    
    summary = {
        "n_sessions": len(sessions),
        "n_presentations": len(session_assignments),
        "mean_coherence": float(np.mean(coherence_values)) if coherence_values else 0.0,
        "std_coherence": float(np.std(coherence_values)) if coherence_values else 0.0,
        "min_coherence": float(np.min(coherence_values)) if coherence_values else 0.0,
        "max_coherence": float(np.max(coherence_values)) if coherence_values else 0.0,
        "mean_fit": float(np.mean(fit_values)) if fit_values else 0.0,
        "std_fit": float(np.std(fit_values)) if fit_values else 0.0,
        "min_session_size": min(m["size"] for m in session_metrics.values()) if session_metrics else 0,
        "max_session_size": max(m["size"] for m in session_metrics.values()) if session_metrics else 0,
    }
    
    return {
        "session_metrics": session_metrics,
        "presentation_metrics": presentation_metrics,
        "session_similarity": session_similarity,
        "summary": summary,
    }


def find_outlier_presentations(
    embeddings: np.ndarray,
    abstract_ids: List[str],
    session_assignments: Dict[str, str],
    fit_threshold: float = 0.5,
    std_threshold: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Find presentations that are poor fits for their sessions.
    
    Uses both absolute fit threshold and relative (z-score) threshold.
    
    Args:
        embeddings: NxD array of embeddings
        abstract_ids: List of abstract IDs
        session_assignments: Dict mapping abstract_id to session_id
        fit_threshold: Absolute fit score threshold
        std_threshold: Number of standard deviations below mean
        
    Returns:
        List of outlier presentations with fit scores and suggestions
    """
    metrics = calculate_all_metrics(embeddings, abstract_ids, session_assignments)
    pres_metrics = metrics["presentation_metrics"]
    session_metrics = metrics["session_metrics"]
    
    # Calculate fit z-scores
    fit_values = [m["fit"] for m in pres_metrics.values()]
    mean_fit = np.mean(fit_values)
    std_fit = np.std(fit_values)
    
    outliers = []
    id_to_idx = {aid: i for i, aid in enumerate(abstract_ids)}
    
    for abstract_id, pm in pres_metrics.items():
        fit = pm["fit"]
        z_score = (fit - mean_fit) / std_fit if std_fit > 0 else 0
        
        is_outlier = fit < fit_threshold or z_score < -std_threshold
        
        if is_outlier:
            idx = id_to_idx.get(abstract_id)
            if idx is None:
                continue
            
            # Find better session matches
            embedding = embeddings[idx:idx+1]
            suggestions = []
            
            for session_id, sm in session_metrics.items():
                if session_id == pm["session_id"]:
                    continue
                
                session_indices = [id_to_idx[aid] for aid in sm["presentation_ids"] 
                                 if aid in id_to_idx]
                if not session_indices:
                    continue
                
                session_embs = embeddings[session_indices]
                potential_fit = calculate_presentation_fit(embedding[0], session_embs, exclude_self=False)
                
                if potential_fit > fit:
                    suggestions.append({
                        "session_id": session_id,
                        "fit": potential_fit,
                        "improvement": potential_fit - fit,
                    })
            
            suggestions.sort(key=lambda x: x["fit"], reverse=True)
            
            outliers.append({
                "abstract_id": abstract_id,
                "current_session": pm["session_id"],
                "fit": fit,
                "z_score": z_score,
                "better_sessions": suggestions[:3],
            })
    
    return sorted(outliers, key=lambda x: x["fit"])


def find_similar_presentations(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    abstract_ids: List[str],
    top_k: int = 10,
    exclude_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Find presentations most similar to a query.
    
    Useful for finding replacement candidates or related presentations.
    
    Args:
        query_embedding: 1xD query embedding
        embeddings: NxD array of all embeddings
        abstract_ids: List of abstract IDs
        top_k: Number of results to return
        exclude_ids: IDs to exclude from results
        
    Returns:
        List of similar presentations with scores
    """
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    exclude_set = set(exclude_ids or [])
    
    results = []
    for i, (aid, sim) in enumerate(zip(abstract_ids, similarities)):
        if aid in exclude_set:
            continue
        results.append({
            "abstract_id": aid,
            "similarity": float(sim),
        })
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

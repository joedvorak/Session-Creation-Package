"""
Placement strategies for session organization.

Provides algorithms for assigning presentations to sessions based on
similarity metrics and constraints.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster


@dataclass
class PlacementResult:
    """Result of a placement operation."""
    session_assignments: Dict[str, str]  # abstract_id -> session_id
    sessions: List[Dict[str, Any]]  # Session metadata
    metadata: Dict[str, Any]  # Algorithm metadata
    unassigned: List[str] = field(default_factory=list)  # Unassigned abstract_ids


@dataclass
class SessionConstraints:
    """Constraints for session creation."""
    min_session_size: int = 8
    max_session_size: int = 12
    max_sessions: Optional[int] = None
    target_session_count: Optional[int] = None


class PlacementStrategy(ABC):
    """
    Abstract base class for placement strategies.
    
    Implementations should handle:
    - Clustering presentations into groups
    - Respecting size constraints
    - Handling pre-assigned (hybrid) sessions
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging/tracking."""
        pass
    
    @abstractmethod
    def place(
        self,
        embeddings: np.ndarray,
        abstract_ids: List[str],
        constraints: SessionConstraints,
        hybrid_assignments: Optional[Dict[str, str]] = None,
    ) -> PlacementResult:
        """
        Assign presentations to sessions.
        
        Args:
            embeddings: NxD array of presentation embeddings
            abstract_ids: List of abstract IDs corresponding to embeddings
            constraints: Session size constraints
            hybrid_assignments: Pre-existing session assignments (abstract_id -> session_id)
            
        Returns:
            PlacementResult with assignments and metadata
        """
        pass


class OralSessionPlacement(PlacementStrategy):
    """
    Hierarchical clustering-based placement for oral sessions (No Hybrid).
    
    Uses a bottom-up traversal of the linkage tree to create sessions.
    The key insight is that as soon as a merge creates a cluster reaching
    the minimum session size, that cluster is "finalized" and set aside.
    This prevents popular topics from being penalized by putting too many
    related presentations in a single oversized session.
    
    NOTE: This strategy does NOT handle hybrid sessions. Pre-assigned
    presentations are simply excluded from clustering. Use HybridFirstPlacement
    if you want hybrid sessions to be filled with similar presentations first.
    
    Algorithm:
    1. Build hierarchical clustering tree from embeddings
    2. Traverse tree bottom-up, tracking unassigned items at each node
    3. When a node reaches min_session_size, finalize it as a session
    4. Continue until max_sessions reached or tree_merge_stop fraction
    5. Assign remaining items to most similar existing sessions
    
    This approach allows popular topics to spawn multiple sessions rather
    than one giant session that gets split or has presentations rejected.
    """
    
    def __init__(
        self,
        linkage_method: str = "average",
        tree_merge_stop: float = 0.95,
        similarity_func=None,
    ):
        """
        Initialize placement strategy.
        
        Args:
            linkage_method: Linkage method for hierarchical clustering
            tree_merge_stop: Fraction of tree to process before stopping (0-1)
            similarity_func: Function to compute similarity (default: cosine_similarity)
        """
        self.linkage_method = linkage_method
        self.tree_merge_stop = tree_merge_stop
        self.similarity_func = similarity_func or cosine_similarity
    
    @property
    def name(self) -> str:
        return "oral_session_hierarchical"
    
    def place(
        self,
        embeddings: np.ndarray,
        abstract_ids: List[str],
        constraints: SessionConstraints,
        hybrid_assignments: Optional[Dict[str, str]] = None,
    ) -> PlacementResult:
        """
        Assign presentations to sessions using bottom-up hierarchical clustering.
        
        Sessions are created as soon as they reach minimum size, allowing
        popular topics to spawn multiple sessions.
        """
        n_items = len(abstract_ids)
        hybrid_assignments = hybrid_assignments or {}
        
        # Build ID mappings
        id_to_idx = {aid: i for i, aid in enumerate(abstract_ids)}
        idx_to_id = {i: aid for i, aid in enumerate(abstract_ids)}
        
        # Identify items already assigned to hybrid sessions
        hybrid_indices = {id_to_idx[aid] for aid in hybrid_assignments if aid in id_to_idx}
        available_indices = [i for i in range(n_items) if i not in hybrid_indices]
        
        # Calculate similarity matrix for assignment phase
        similarity_matrix = self.similarity_func(embeddings, embeddings)
        
        # Get max sessions (accounting for hybrid sessions)
        n_hybrid_sessions = len(set(hybrid_assignments.values()))
        if constraints.max_sessions:
            max_new_sessions = constraints.max_sessions - n_hybrid_sessions
        else:
            max_new_sessions = len(available_indices) // constraints.min_session_size
        
        max_new_sessions = max(1, max_new_sessions)
        
        # Perform bottom-up clustering on available items
        if len(available_indices) < 2:
            final_clusters = []
            if available_indices:
                final_clusters = [available_indices]
        else:
            final_clusters = self._bottom_up_cluster(
                embeddings=embeddings,
                available_indices=available_indices,
                similarity_matrix=similarity_matrix,
                min_session_size=constraints.min_session_size,
                max_session_size=constraints.max_session_size,
                max_sessions=max_new_sessions,
            )
        
        # Create session assignments
        session_assignments = dict(hybrid_assignments)
        sessions = []
        
        # Add hybrid sessions first
        hybrid_session_ids = set(hybrid_assignments.values())
        for session_id in hybrid_session_ids:
            pres_ids = [aid for aid, sid in hybrid_assignments.items() if sid == session_id]
            sessions.append({
                "session_id": session_id,
                "is_hybrid": True,
                "presentation_ids": pres_ids,
                "size": len(pres_ids),
            })
        
        # Add new sessions
        for cluster_id, indices in enumerate(final_clusters):
            session_id = f"SESSION-{cluster_id + 1:03d}"
            pres_ids = [idx_to_id[idx] for idx in indices]
            
            for aid in pres_ids:
                session_assignments[aid] = session_id
            
            sessions.append({
                "session_id": session_id,
                "is_hybrid": False,
                "presentation_ids": pres_ids,
                "size": len(pres_ids),
            })
        
        # Identify unassigned
        unassigned = [aid for aid in abstract_ids if aid not in session_assignments]
        
        metadata = {
            "strategy": self.name,
            "n_items": n_items,
            "n_hybrid": len(hybrid_indices),
            "n_available": len(available_indices),
            "n_sessions": len(sessions),
            "n_assigned": len(session_assignments),
            "n_unassigned": len(unassigned),
            "tree_merge_stop": self.tree_merge_stop,
            "linkage_method": self.linkage_method,
        }
        
        return PlacementResult(
            session_assignments=session_assignments,
            sessions=sessions,
            metadata=metadata,
            unassigned=unassigned,
        )
    
    def _bottom_up_cluster(
        self,
        embeddings: np.ndarray,
        available_indices: List[int],
        similarity_matrix: np.ndarray,
        min_session_size: int,
        max_session_size: int,
        max_sessions: int,
    ) -> List[List[int]]:
        """
        Traverse linkage tree bottom-up, finalizing sessions when they reach target size.
        
        This is the key algorithm that prevents popular topics from being penalized.
        As merges happen, we track unassigned items. When a merge creates a group
        >= min_session_size, we finalize it and remove those items from consideration.
        """
        n_available = len(available_indices)
        
        # Create mapping from filtered position to original index
        filtered_to_original = {i: available_indices[i] for i in range(n_available)}
        
        # Extract embeddings for available items and build linkage
        filtered_embeddings = embeddings[available_indices]
        
        linkage_matrix = linkage(
            y=filtered_embeddings,
            method=self.linkage_method,
            metric='cosine',
        )
        
        n_nodes = linkage_matrix.shape[0]
        merge_stop_index = int(n_nodes * self.tree_merge_stop)
        
        # Track unassigned count and leaves at each internal node
        # unassigned_count[i] = number of unassigned items under node (n_available + i)
        unassigned_count = linkage_matrix[:, 3].copy()  # Start with total count per node
        unassigned_leaves = [[] for _ in range(n_nodes)]
        
        final_clusters = []
        
        # Process tree bottom-up
        for i in range(n_nodes):
            left_child = int(linkage_matrix[i, 0])
            right_child = int(linkage_matrix[i, 1])
            
            # Collect leaves from left child
            if left_child >= n_available:
                # Internal node
                left_internal = left_child - n_available
                left_size = unassigned_count[left_internal]
                unassigned_leaves[i].extend(unassigned_leaves[left_internal])
            else:
                # Leaf node (filtered position)
                left_size = 1
                unassigned_leaves[i].append(left_child)
            
            # Collect leaves from right child
            if right_child >= n_available:
                # Internal node
                right_internal = right_child - n_available
                right_size = unassigned_count[right_internal]
                unassigned_leaves[i].extend(unassigned_leaves[right_internal])
            else:
                # Leaf node (filtered position)
                right_size = 1
                unassigned_leaves[i].append(right_child)
            
            # Update count for this node
            unassigned_count[i] = left_size + right_size
            
            # Check if we should finalize this cluster.
            # Stop creating sessions once merge_stop OR max_sessions is reached.
            # This prevents tail-overloading (PLACE-001) by ensuring all
            # remaining items are distributed to existing sessions.
            should_finalize = (
                unassigned_count[i] >= min_session_size and 
                i < merge_stop_index and 
                len(final_clusters) < max_sessions
            )
            
            if should_finalize:
                
                # Convert filtered positions to original indices
                original_indices = [filtered_to_original[pos] for pos in unassigned_leaves[i]]
                final_clusters.append(original_indices)
                
                # Mark as assigned (clear from tracking)
                unassigned_count[i] = 0
                unassigned_leaves[i] = []
        
        # Handle remaining unassigned items
        remaining_positions = unassigned_leaves[-1] if n_nodes > 0 else list(range(n_available))
        
        if remaining_positions:
            remaining_original = [filtered_to_original[pos] for pos in remaining_positions]
            
            if final_clusters:
                # Assign to most similar existing sessions (respecting max_session_size)
                final_clusters = self._assign_remaining_items(
                    remaining_original, final_clusters, similarity_matrix, max_session_size
                )
            else:
                # No sessions created yet - make one from remaining
                final_clusters.append(remaining_original)
        
        return final_clusters
    
    def _assign_remaining_items(
        self,
        remaining_indices: List[int],
        final_clusters: List[List[int]],
        similarity_matrix: np.ndarray,
        max_session_size: int = 12,
    ) -> List[List[int]]:
        """
        Assign remaining items to most similar existing clusters.
        
        No new sessions are created here. Each remaining item goes to
        the cluster with the highest average similarity, regardless of
        max_session_size. This matches the legacy behavior and prevents
        tail-overloading (PLACE-001) where single-item overflow sessions
        became magnets for subsequent items.
        """
        if not remaining_indices or not final_clusters:
            if remaining_indices:
                final_clusters.append(remaining_indices)
            return final_clusters
        
        for idx in remaining_indices:
            best_cluster_idx = 0
            best_similarity = -1
            
            # Find cluster with highest average similarity
            for cluster_idx, cluster_indices in enumerate(final_clusters):
                if cluster_indices:
                    similarities = [similarity_matrix[idx, ci] for ci in cluster_indices]
                    avg_similarity = np.mean(similarities)
                    
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_cluster_idx = cluster_idx
            
            final_clusters[best_cluster_idx].append(idx)
        
        return final_clusters


class HybridFirstPlacement(PlacementStrategy):
    """
    Hierarchical clustering-based placement that fills hybrid sessions first.
    
    This strategy implements a two-phase approach:
    1. HYBRID FILL PHASE: Hybrid sessions get first pick on the most similar
       presentations from the general pool to reach minimum session size
    2. CLUSTERING PHASE: Remaining presentations are clustered using bottom-up
       hierarchical clustering (same as OralSessionPlacement)
    
    This ensures that invited/special sessions are filled with relevant content
    before the general clustering begins.
    
    Algorithm:
    1. For each hybrid session, compute representative embedding (mean)
    2. Calculate similarity between hybrid sessions and all general presentations
    3. Assign most similar presentations to each hybrid session to reach min_size
    4. Build hierarchical clustering tree from remaining embeddings
    5. Traverse tree bottom-up, finalizing sessions at target sizes
    6. Assign remaining items to most similar existing sessions
    """
    
    def __init__(
        self,
        linkage_method: str = "average",
        tree_merge_stop: float = 0.95,
        similarity_func=None,
    ):
        """
        Initialize placement strategy.
        
        Args:
            linkage_method: Linkage method for hierarchical clustering
            tree_merge_stop: Fraction of tree to process before stopping (0-1)
            similarity_func: Function to compute similarity (default: cosine_similarity)
        """
        self.linkage_method = linkage_method
        self.tree_merge_stop = tree_merge_stop
        self.similarity_func = similarity_func or cosine_similarity
    
    @property
    def name(self) -> str:
        return "hybrid_first_hierarchical"
    
    def place(
        self,
        embeddings: np.ndarray,
        abstract_ids: List[str],
        constraints: SessionConstraints,
        hybrid_assignments: Optional[Dict[str, str]] = None,
        hybrid_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> PlacementResult:
        """
        Assign presentations to sessions, filling hybrid sessions first.
        
        Args:
            embeddings: NxD array of presentation embeddings
            abstract_ids: List of abstract IDs corresponding to embeddings
            constraints: Session size constraints
            hybrid_assignments: Pre-existing session assignments (abstract_id -> session_id)
            hybrid_embeddings: Embeddings for hybrid presentations (abstract_id -> embedding)
                              If not provided, will use embeddings from main array
            
        Returns:
            PlacementResult with assignments and metadata
        """
        n_items = len(abstract_ids)
        hybrid_assignments = hybrid_assignments or {}
        
        # Build ID mappings
        id_to_idx = {aid: i for i, aid in enumerate(abstract_ids)}
        idx_to_id = {i: aid for i, aid in enumerate(abstract_ids)}
        
        # Calculate similarity matrix for all presentations
        similarity_matrix = self.similarity_func(embeddings, embeddings)
        
        # Track assigned indices
        assigned_indices: Set[int] = set()
        
        # Structure to hold final sessions
        # Each entry: {"session_id": str, "indices": List[int], "is_hybrid": bool, "hybrid_pres_ids": List[str]}
        sessions_data: List[Dict[str, Any]] = []
        
        # ========== PHASE 1: Fill Hybrid Sessions ==========
        if hybrid_assignments:
            # Group hybrid presentations by session
            session_to_hybrid_ids: Dict[str, List[str]] = {}
            for aid, sid in hybrid_assignments.items():
                if sid not in session_to_hybrid_ids:
                    session_to_hybrid_ids[sid] = []
                session_to_hybrid_ids[sid].append(aid)
            
            # For each hybrid session, compute representative embedding and fill
            for session_id, hybrid_pres_ids in session_to_hybrid_ids.items():
                # Get embeddings for hybrid presentations in this session
                hybrid_embs = []
                for aid in hybrid_pres_ids:
                    if hybrid_embeddings and aid in hybrid_embeddings:
                        hybrid_embs.append(hybrid_embeddings[aid])
                    elif aid in id_to_idx:
                        # Use embedding from main array
                        hybrid_embs.append(embeddings[id_to_idx[aid]])
                
                if not hybrid_embs:
                    # No embeddings available, just track the hybrid session
                    sessions_data.append({
                        "session_id": session_id,
                        "indices": [],
                        "is_hybrid": True,
                        "hybrid_pres_ids": hybrid_pres_ids,
                    })
                    continue
                
                # Compute representative embedding (mean)
                representative_emb = np.mean(hybrid_embs, axis=0).reshape(1, -1)
                
                # Calculate how many more presentations needed
                current_size = len(hybrid_pres_ids)
                need_count = max(0, constraints.min_session_size - current_size)
                
                # Calculate similarity to all general presentations
                # Only consider presentations not yet assigned
                available_indices = [i for i in range(n_items) 
                                    if i not in assigned_indices 
                                    and idx_to_id[i] not in hybrid_assignments]
                
                if need_count > 0 and available_indices:
                    # Get similarities between representative and available presentations
                    available_embs = embeddings[available_indices]
                    similarities = self.similarity_func(representative_emb, available_embs)[0]
                    
                    # Get top-k most similar
                    k = min(need_count, len(available_indices))
                    if k > 0:
                        top_k_local = np.argpartition(similarities, -k)[-k:]
                        top_k_local = top_k_local[np.argsort(similarities[top_k_local])[::-1]]
                        
                        # Convert to original indices
                        selected_indices = [available_indices[i] for i in top_k_local]
                        assigned_indices.update(selected_indices)
                        
                        sessions_data.append({
                            "session_id": session_id,
                            "indices": selected_indices,
                            "is_hybrid": True,
                            "hybrid_pres_ids": hybrid_pres_ids,
                        })
                    else:
                        sessions_data.append({
                            "session_id": session_id,
                            "indices": [],
                            "is_hybrid": True,
                            "hybrid_pres_ids": hybrid_pres_ids,
                        })
                else:
                    sessions_data.append({
                        "session_id": session_id,
                        "indices": [],
                        "is_hybrid": True,
                        "hybrid_pres_ids": hybrid_pres_ids,
                    })
        
        # ========== PHASE 2: Cluster Remaining Presentations ==========
        # Get indices of presentations still available
        available_indices = [i for i in range(n_items) 
                           if i not in assigned_indices 
                           and idx_to_id[i] not in hybrid_assignments]
        
        # Calculate max sessions for new clusters
        n_hybrid_sessions = len(set(hybrid_assignments.values())) if hybrid_assignments else 0
        if constraints.max_sessions:
            max_new_sessions = constraints.max_sessions - n_hybrid_sessions
        else:
            max_new_sessions = len(available_indices) // constraints.min_session_size
        
        max_new_sessions = max(1, max_new_sessions)
        
        # Perform bottom-up clustering on remaining items
        if len(available_indices) >= 2:
            new_clusters = self._bottom_up_cluster(
                embeddings=embeddings,
                available_indices=available_indices,
                similarity_matrix=similarity_matrix,
                min_session_size=constraints.min_session_size,
                max_session_size=constraints.max_session_size,
                max_sessions=max_new_sessions,
            )
        elif available_indices:
            new_clusters = [available_indices]
        else:
            new_clusters = []
        
        # ========== Build Final Result ==========
        session_assignments = {}
        sessions = []
        
        # Add hybrid sessions
        for session_data in sessions_data:
            session_id = session_data["session_id"]
            hybrid_pres_ids = session_data["hybrid_pres_ids"]
            filled_indices = session_data["indices"]
            
            # Combine hybrid presentations with filled presentations
            all_pres_ids = list(hybrid_pres_ids) + [idx_to_id[idx] for idx in filled_indices]
            
            for aid in all_pres_ids:
                session_assignments[aid] = session_id
            
            sessions.append({
                "session_id": session_id,
                "is_hybrid": True,
                "presentation_ids": all_pres_ids,
                "size": len(all_pres_ids),
                "hybrid_original_ids": hybrid_pres_ids,
                "filled_from_general": [idx_to_id[idx] for idx in filled_indices],
            })
        
        # Add new clustered sessions
        for cluster_id, indices in enumerate(new_clusters):
            session_id = f"SESSION-{cluster_id + 1:03d}"
            pres_ids = [idx_to_id[idx] for idx in indices]
            
            for aid in pres_ids:
                session_assignments[aid] = session_id
            
            sessions.append({
                "session_id": session_id,
                "is_hybrid": False,
                "presentation_ids": pres_ids,
                "size": len(pres_ids),
            })
        
        # Identify unassigned
        unassigned = [aid for aid in abstract_ids if aid not in session_assignments]
        
        metadata = {
            "strategy": self.name,
            "n_items": n_items,
            "n_hybrid_sessions": n_hybrid_sessions,
            "n_hybrid_presentations": len(hybrid_assignments),
            "n_filled_to_hybrid": sum(len(s.get("filled_from_general", [])) for s in sessions if s["is_hybrid"]),
            "n_available_for_clustering": len(available_indices),
            "n_new_sessions": len(new_clusters),
            "n_total_sessions": len(sessions),
            "n_assigned": len(session_assignments),
            "n_unassigned": len(unassigned),
            "tree_merge_stop": self.tree_merge_stop,
            "linkage_method": self.linkage_method,
        }
        
        return PlacementResult(
            session_assignments=session_assignments,
            sessions=sessions,
            metadata=metadata,
            unassigned=unassigned,
        )
    
    def _bottom_up_cluster(
        self,
        embeddings: np.ndarray,
        available_indices: List[int],
        similarity_matrix: np.ndarray,
        min_session_size: int,
        max_session_size: int,
        max_sessions: int,
    ) -> List[List[int]]:
        """
        Traverse linkage tree bottom-up, finalizing sessions when they reach target size.
        
        Same algorithm as OralSessionPlacement._bottom_up_cluster.
        """
        n_available = len(available_indices)
        
        # Create mapping from filtered position to original index
        filtered_to_original = {i: available_indices[i] for i in range(n_available)}
        
        # Extract embeddings for available items and build linkage
        filtered_embeddings = embeddings[available_indices]
        
        linkage_matrix = linkage(
            y=filtered_embeddings,
            method=self.linkage_method,
            metric='cosine',
        )
        
        n_nodes = linkage_matrix.shape[0]
        merge_stop_index = int(n_nodes * self.tree_merge_stop)
        
        # Track unassigned count and leaves at each internal node
        unassigned_count = linkage_matrix[:, 3].copy()
        unassigned_leaves = [[] for _ in range(n_nodes)]
        
        final_clusters = []
        
        # Process tree bottom-up
        for i in range(n_nodes):
            left_child = int(linkage_matrix[i, 0])
            right_child = int(linkage_matrix[i, 1])
            
            # Collect leaves from left child
            if left_child >= n_available:
                left_internal = left_child - n_available
                left_size = unassigned_count[left_internal]
                unassigned_leaves[i].extend(unassigned_leaves[left_internal])
            else:
                left_size = 1
                unassigned_leaves[i].append(left_child)
            
            # Collect leaves from right child
            if right_child >= n_available:
                right_internal = right_child - n_available
                right_size = unassigned_count[right_internal]
                unassigned_leaves[i].extend(unassigned_leaves[right_internal])
            else:
                right_size = 1
                unassigned_leaves[i].append(right_child)
            
            unassigned_count[i] = left_size + right_size
            
            # Check if we should finalize this cluster.
            # Stop creating sessions once merge_stop OR max_sessions is reached.
            should_finalize = (
                unassigned_count[i] >= min_session_size and 
                i < merge_stop_index and 
                len(final_clusters) < max_sessions
            )
            
            if should_finalize:
                original_indices = [filtered_to_original[pos] for pos in unassigned_leaves[i]]
                final_clusters.append(original_indices)
                unassigned_count[i] = 0
                unassigned_leaves[i] = []
        
        # Handle remaining unassigned items
        remaining_positions = unassigned_leaves[-1] if n_nodes > 0 else list(range(n_available))
        
        if remaining_positions:
            remaining_original = [filtered_to_original[pos] for pos in remaining_positions]
            
            if final_clusters:
                final_clusters = self._assign_remaining_items(
                    remaining_original, final_clusters, similarity_matrix, max_session_size
                )
            else:
                final_clusters.append(remaining_original)
        
        return final_clusters
    
    def _assign_remaining_items(
        self,
        remaining_indices: List[int],
        final_clusters: List[List[int]],
        similarity_matrix: np.ndarray,
        max_session_size: int = 12,
    ) -> List[List[int]]:
        """
        Assign remaining items to most similar existing clusters.
        
        No new sessions are created here. Each remaining item goes to
        the cluster with the highest average similarity, regardless of
        max_session_size. This matches the legacy behavior and prevents
        tail-overloading (PLACE-001).
        """
        if not remaining_indices or not final_clusters:
            if remaining_indices:
                final_clusters.append(remaining_indices)
            return final_clusters
        
        for idx in remaining_indices:
            best_cluster_idx = 0
            best_similarity = -1
            
            for cluster_idx, cluster_indices in enumerate(final_clusters):
                if cluster_indices:
                    similarities = [similarity_matrix[idx, ci] for ci in cluster_indices]
                    avg_similarity = np.mean(similarities)
                    
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_cluster_idx = cluster_idx
            
            final_clusters[best_cluster_idx].append(idx)
        
        return final_clusters


class TraditionalClusterPlacement(PlacementStrategy):
    """
    Traditional fcluster-based placement for comparison/research purposes.
    
    Uses scipy's fcluster to cut the hierarchical tree at a fixed level,
    creating a predetermined number of clusters. This is a more standard
    approach but may penalize popular topics by creating one large cluster
    that then needs to be split.
    
    This strategy is provided for comparison with OralSessionPlacement.
    For production use, OralSessionPlacement is recommended.
    """
    
    def __init__(
        self,
        linkage_method: str = "average",
        tree_merge_stop: float = 0.95,  # Accepted but not used (for API compatibility)
        similarity_func=None,
    ):
        self.linkage_method = linkage_method
        self.tree_merge_stop = tree_merge_stop  # Not used in this strategy
        self.similarity_func = similarity_func or cosine_similarity
    
    @property
    def name(self) -> str:
        return "traditional_fcluster"
    
    def place(
        self,
        embeddings: np.ndarray,
        abstract_ids: List[str],
        constraints: SessionConstraints,
        hybrid_assignments: Optional[Dict[str, str]] = None,
    ) -> PlacementResult:
        """
        Assign presentations using traditional fcluster approach.
        """
        n_items = len(abstract_ids)
        hybrid_assignments = hybrid_assignments or {}
        
        id_to_idx = {aid: i for i, aid in enumerate(abstract_ids)}
        idx_to_id = {i: aid for i, aid in enumerate(abstract_ids)}
        
        hybrid_indices = {id_to_idx[aid] for aid in hybrid_assignments if aid in id_to_idx}
        available_list = [i for i in range(n_items) if i not in hybrid_indices]
        
        similarity_matrix = self.similarity_func(embeddings, embeddings)
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        if len(available_list) < 2:
            session_assignments = dict(hybrid_assignments)
            sessions = []
            for session_id in set(hybrid_assignments.values()):
                pres_ids = [aid for aid, sid in hybrid_assignments.items() if sid == session_id]
                sessions.append({"session_id": session_id, "is_hybrid": True, 
                               "presentation_ids": pres_ids, "size": len(pres_ids)})
            return PlacementResult(
                session_assignments=session_assignments,
                sessions=sessions,
                metadata={"strategy": self.name, "n_items": n_items},
                unassigned=[],
            )
        
        # Build linkage tree
        filtered_embeddings = embeddings[available_list]
        Z = linkage(filtered_embeddings, method=self.linkage_method, metric='cosine')
        
        # Determine target number of clusters
        n_hybrid_sessions = len(set(hybrid_assignments.values()))
        if constraints.max_sessions:
            n_clusters = constraints.max_sessions - n_hybrid_sessions
        else:
            n_clusters = len(available_list) // constraints.min_session_size
        
        n_clusters = max(1, n_clusters)
        
        # Cut tree
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Group by cluster
        clusters: Dict[int, List[int]] = {}
        for i, label in enumerate(labels):
            orig_idx = available_list[i]
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(orig_idx)
        
        # Refine clusters (merge small, split large)
        final_clusters = self._refine_clusters(
            clusters, similarity_matrix, constraints
        )
        
        # Build result
        session_assignments = dict(hybrid_assignments)
        sessions = []
        
        for session_id in set(hybrid_assignments.values()):
            pres_ids = [aid for aid, sid in hybrid_assignments.items() if sid == session_id]
            sessions.append({"session_id": session_id, "is_hybrid": True,
                           "presentation_ids": pres_ids, "size": len(pres_ids)})
        
        for cluster_id, (_, indices) in enumerate(final_clusters.items()):
            session_id = f"SESSION-{cluster_id + 1:03d}"
            pres_ids = [idx_to_id[idx] for idx in indices]
            for aid in pres_ids:
                session_assignments[aid] = session_id
            sessions.append({"session_id": session_id, "is_hybrid": False,
                           "presentation_ids": pres_ids, "size": len(pres_ids)})
        
        unassigned = [aid for aid in abstract_ids if aid not in session_assignments]
        
        return PlacementResult(
            session_assignments=session_assignments,
            sessions=sessions,
            metadata={"strategy": self.name, "n_items": n_items, "n_sessions": len(sessions)},
            unassigned=unassigned,
        )
    
    def _refine_clusters(
        self,
        clusters: Dict[int, List[int]],
        similarity_matrix: np.ndarray,
        constraints: SessionConstraints,
    ) -> Dict[int, List[int]]:
        """Merge small clusters and split oversized ones."""
        min_size = constraints.min_session_size
        max_size = constraints.max_session_size
        
        # Merge small clusters
        changed = True
        while changed:
            changed = False
            small = [cid for cid, idx in clusters.items() if len(idx) < min_size]
            
            for small_cid in small:
                if small_cid not in clusters:
                    continue
                small_indices = clusters[small_cid]
                
                best_target, best_sim = None, -1
                for target_cid, target_indices in clusters.items():
                    if target_cid == small_cid:
                        continue
                    if len(target_indices) + len(small_indices) > max_size * 1.5:
                        continue
                    
                    sim = np.mean([similarity_matrix[i, j] 
                                  for i in small_indices for j in target_indices])
                    if sim > best_sim:
                        best_sim, best_target = sim, target_cid
                
                if best_target is not None:
                    clusters[best_target].extend(small_indices)
                    del clusters[small_cid]
                    changed = True
        
        # Split oversized
        next_id = max(clusters.keys()) + 1 if clusters else 1
        for cid in list(clusters.keys()):
            while len(clusters[cid]) > max_size * 1.5 and len(clusters[cid]) >= 2 * min_size:
                indices = clusters[cid]
                # Find two most distant
                min_sim, seed1, seed2 = float('inf'), indices[0], indices[1]
                for i, idx1 in enumerate(indices):
                    for idx2 in indices[i+1:]:
                        if similarity_matrix[idx1, idx2] < min_sim:
                            min_sim = similarity_matrix[idx1, idx2]
                            seed1, seed2 = idx1, idx2
                
                g1, g2 = [seed1], [seed2]
                for idx in indices:
                    if idx in (seed1, seed2):
                        continue
                    if similarity_matrix[idx, seed1] > similarity_matrix[idx, seed2]:
                        g1.append(idx)
                    else:
                        g2.append(idx)
                
                clusters[cid] = g1
                clusters[next_id] = g2
                next_id += 1
        
        # Handle remaining small clusters
        final = {cid: idx for cid, idx in clusters.items() if len(idx) >= min_size}
        orphans = [i for cid, idx in clusters.items() if len(idx) < min_size for i in idx]
        
        for orphan in orphans:
            best_cid = max(final.keys(), key=lambda c: np.mean(
                [similarity_matrix[orphan, i] for i in final[c]]) if final[c] else -1)
            final[best_cid].append(orphan)
        
        return {i: idx for i, (_, idx) in enumerate(final.items(), 1)}


class PosterThematicOrdering(PlacementStrategy):
    """
    Order poster presentations by thematic similarity.
    
    Creates a linear ordering where similar posters are adjacent,
    useful for physical poster session layout.
    """
    
    def __init__(
        self, 
        similarity_func=None,
        linkage_method: str = "average",  # Accepted but not used
        tree_merge_stop: float = 0.95,  # Accepted but not used
    ):
        self.similarity_func = similarity_func or cosine_similarity
        # linkage_method and tree_merge_stop not used for poster ordering
    
    @property
    def name(self) -> str:
        return "poster_thematic_ordering"
    
    def place(
        self,
        embeddings: np.ndarray,
        abstract_ids: List[str],
        constraints: SessionConstraints,
        hybrid_assignments: Optional[Dict[str, str]] = None,
    ) -> PlacementResult:
        """
        Create thematic ordering using traveling salesman approximation.
        
        All posters go in one "session" but with order_index assigned.
        """
        n_items = len(abstract_ids)
        
        # Calculate similarity matrix
        similarity_matrix = self.similarity_func(embeddings, embeddings)
        
        # Use nearest-neighbor heuristic for ordering
        order = self._nearest_neighbor_order(similarity_matrix)
        
        # Create single poster session with ordering
        session_id = "POSTER-001"
        session_assignments = {abstract_ids[i]: session_id for i in range(n_items)}
        
        # Store order in metadata
        ordered_ids = [abstract_ids[i] for i in order]
        
        sessions = [{
            "session_id": session_id,
            "is_hybrid": False,
            "presentation_ids": ordered_ids,
            "size": n_items,
            "ordering": {abstract_ids[order[i]]: i for i in range(n_items)},
        }]
        
        return PlacementResult(
            session_assignments=session_assignments,
            sessions=sessions,
            metadata={
                "strategy": self.name,
                "n_items": n_items,
                "order": ordered_ids,
            },
            unassigned=[],
        )
    
    def _nearest_neighbor_order(self, similarity_matrix: np.ndarray) -> List[int]:
        """
        Create ordering using nearest-neighbor heuristic.
        
        Start with random item, always go to most similar unvisited item.
        """
        n = similarity_matrix.shape[0]
        if n == 0:
            return []
        
        visited = [False] * n
        order = []
        
        # Start with item 0
        current = 0
        order.append(current)
        visited[current] = True
        
        while len(order) < n:
            # Find most similar unvisited
            best_next = None
            best_sim = -1
            
            for j in range(n):
                if not visited[j]:
                    sim = similarity_matrix[current, j]
                    if sim > best_sim:
                        best_sim = sim
                        best_next = j
            
            if best_next is not None:
                order.append(best_next)
                visited[best_next] = True
                current = best_next
            else:
                break
        
        return order


def create_placement_strategy(
    strategy_type: str = "oral",
    **kwargs
) -> PlacementStrategy:
    """
    Factory function to create placement strategies.
    
    Args:
        strategy_type: Type of strategy:
            - 'oral': Bottom-up hierarchical clustering, no hybrid fill (recommended for oral sessions)
            - 'hybrid_first': Bottom-up hierarchical with hybrid sessions filled first
            - 'traditional': Traditional fcluster approach (for comparison/research)
            - 'poster_thematic': Thematic ordering for poster sessions
        **kwargs: Strategy-specific arguments
        
    Returns:
        PlacementStrategy instance
    """
    if strategy_type == "oral":
        return OralSessionPlacement(**kwargs)
    elif strategy_type == "hybrid_first":
        return HybridFirstPlacement(**kwargs)
    elif strategy_type == "traditional":
        return TraditionalClusterPlacement(**kwargs)
    elif strategy_type in ("poster", "poster_thematic"):
        return PosterThematicOrdering(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

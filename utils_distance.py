import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra


def compute_geodesic_distances(V, F, fixed_vertices):
    """
    Computes geodesic distances from each vertex to the fixed vertices.
    
    Parameters:
    - V: np.array of shape (N, 3) containing the vertices of the mesh.
    - F: np.array of shape (M, 3) containing the faces of the mesh.
    - fixed_vertices: np.array of indices of fixed vertices.
    
    Returns:
    - distances: np.array of shape (num_fixed, N) containing the geodesic distances from each vertex to the fixed vertices.
    """
    N = V.shape[0]
    distances = np.full((len(fixed_vertices), N), np.inf)
    
    graph = csr_matrix((N, N))
    for face in F:
        for i in range(3):
            i0, i1 = face[i], face[(i + 1) % 3]
            dist = np.linalg.norm(V[i0] - V[i1])
            graph[i0, i1] = dist
            graph[i1, i0] = dist
    
    for idx, fv in enumerate(fixed_vertices):
        distances[idx], _ = dijkstra(graph, indices=fv, return_predecessors=True)
    
    return distances



def compute_euclidean_distances( V, fixed_vertices ):
    """
    The same thig but computing euclidean distances.
    """
    qty_fixed = len(fixed_vertices)
    qty_total = V.shape[0]
    distances = np.zeros( (qty_fixed, qty_total) )
    for idx, abs_idx in enumerate(fixed_vertices):
        V_fixed = V[abs_idx]
        dists = np.linalg.norm(V.T - V_fixed[:, None], axis=0)
        distances[idx] = dists
    return distances





def compute_distances(Vs, Fs, selected_vertices, metric_types):
    """
    Computes distances from selected vertices to all other vertices using specified metrics.

    Parameters:
        Vs (ndarray): N x 3 array of vertex positions.
        Fs (ndarray): M x 3 array of face indices.
        selected_vertices (list): List of selected vertex indices.
        metric_types (list): List of strings "geodesic" or "euclidean" corresponding to selected_vertices.

    Returns:
        ndarray: Distances for all selected vertices, in the same order as selected_vertices.
    """
    # Validate input
    if len(selected_vertices) != len(metric_types):
        raise ValueError("Length of selected_vertices must match length of metric_types.")
    
    # Separate selected vertices by metric type
    geodesic_indices = [i for i, m in enumerate(metric_types) if m == "geodesic"]
    euclidean_indices = [i for i, m in enumerate(metric_types) if m == "euclidean"]
    
    geodesic_vertices = [selected_vertices[i] for i in geodesic_indices]
    euclidean_vertices = [selected_vertices[i] for i in euclidean_indices]
    
    # Compute distances for each type
    geodesic_distances = (
        compute_geodesic_distances(Vs, Fs, geodesic_vertices)
        if geodesic_vertices else np.empty((0, Vs.shape[0]))
    )
    euclidean_distances = (
        compute_euclidean_distances(Vs, euclidean_vertices)
        if euclidean_vertices else np.empty((0, Vs.shape[0]))
    )
    
    # Combine results in the original order
    all_distances = []
    geo_idx, eu_idx = 0, 0
    for metric in metric_types:
        if metric == "geodesic":
            all_distances.append(geodesic_distances[geo_idx])
            geo_idx += 1
        elif metric == "euclidean":
            all_distances.append(euclidean_distances[eu_idx])
            eu_idx += 1
    
    return np.array(all_distances)





def extract_reachable_vertices(Vs, Fs, selected_vertices, selected_positions, distances):
    """
    Re-pack Vs, Fs, selected_vertices, and distances to only include reachable vertices.

    Parameters:
        Vs (ndarray): N x 3 array of vertex positions.
        Fs (ndarray): M x 3 array of face indices.
        selected_vertices (list): List of selected vertex indices.
        distances (ndarray): Array of distances from the compute_distances() function.

    Returns:
        tuple: (new_Vs, new_Fs, new_selected_vertices, new_distances, reachable_indices)
            - new_Vs: Filtered vertex positions.
            - new_Fs: Filtered face indices.
            - new_selected_vertices: Filtered selected vertices.
            - new_distances: Filtered distances.
            - reachable_indices: Indices of reachable vertices in the original Vs array.
    """
    # Determine reachable vertices (non-negative distances for at least one selected vertex)
    reachable_mask = np.any(distances >= 0, axis=0)
    reachable_indices = np.where(reachable_mask)[0]
    unreachable_indices = np.where(~reachable_mask)[0]

    # Map old indices to new indices
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(reachable_indices)}

    # Filter Vs
    new_Vs = Vs[reachable_indices]
    unreachable_Vs = Vs[unreachable_indices]

    # Filter Fs: Keep only faces where all vertices are reachable
    new_Fs = []
    for face in Fs:
        if all(vertex in reachable_indices for vertex in face):
            new_Fs.append([index_map[v] for v in face])
    new_Fs = np.array(new_Fs)

    # Filter distances
    new_distances = distances[:, reachable_indices]

    return unreachable_Vs, unreachable_indices, new_Vs, new_Fs, new_distances, reachable_indices




def update_original_vertices( original_Vs, modified_reachable_Vs, reachable_indices ):
    """
    Updates the original vertex array with the modified reachable vertices.

    Parameters:
        original_Vs (ndarray): N x 3 array of the original vertex positions.
        modified_reachable_Vs (ndarray): K x 3 array of modified reachable vertex positions.
        reachable_indices (ndarray): Indices of the reachable vertices in the original array.

    Returns:
        ndarray: Updated vertex positions for the original array.
    """
    # Create a copy of the original vertices to avoid modifying in place
    updated_Vs = original_Vs.copy()
    
    # Update the reachable vertices in the original array
    updated_Vs[reachable_indices] = modified_reachable_Vs
    
    return updated_Vs




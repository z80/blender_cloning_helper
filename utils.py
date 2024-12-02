import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra

def compute_cotangent_weights(V, F):
    """
    Computes cotangent weights for each edge in the mesh.
    
    Parameters:
    - V: np.array of shape (N, 3) containing the vertices of the mesh.
    - F: np.array of shape (M, 3) containing the faces of the mesh.
    
    Returns:
    - weights: A dictionary where keys are edges (i, j) and values are the computed cotangent weights.
    """
    weights = {}
    for face in F:
        for i in range(3):
            i0, i1, i2 = face[i], face[(i + 1) % 3], face[(i + 2) % 3]
            v0, v1, v2 = V[i0], V[i1], V[i2]
            edge1, edge2 = v1 - v0, v2 - v0
            angle = np.arccos(np.clip(np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2)), -1.0, 1.0))
            cot_angle = 1 / np.tan(angle)
            weights[(i0, i1)] = weights.get((i0, i1), 0) + 0.5 * cot_angle
            weights[(i1, i0)] = weights.get((i1, i0), 0) + 0.5 * cot_angle
            weights[(i1, i2)] = weights.get((i1, i2), 0) + 0.5 * cot_angle
            weights[(i2, i1)] = weights.get((i2, i1), 0) + 0.5 * cot_angle
            weights[(i2, i0)] = weights.get((i2, i0), 0) + 0.5 * cot_angle
            weights[(i0, i2)] = weights.get((i0, i2), 0) + 0.5 * cot_angle
    return weights

def arap(V, F, fixed_vertices, fixed_positions, iterations=10):
    """
    Executes the As-Rigid-As-Possible (ARAP) optimization.
    
    Parameters:
    - V: np.array of shape (N, 3) containing the vertices of the mesh.
    - F: np.array of shape (M, 3) containing the faces of the mesh.
    - fixed_vertices: np.array of indices of vertices that are fixed.
    - fixed_positions: np.array of shape (num_fixed, 3) containing the positions of the fixed vertices.
    - iterations: int, number of iterations for the optimization process (default is 10).
    
    Returns:
    - V_new: np.array of shape (N, 3) containing the new vertex positions after ARAP optimization.
    """
    # Compute cotangent weights
    cotangent_weights = compute_cotangent_weights(V, F)
    
    # Initialize variables
    N = V.shape[0]
    V_new = V.copy()
    
    for iter in range(iterations):
        # Step 1: Compute centroids of faces
        centroids = np.mean(V_new[F], axis=1)
        
        # Step 2: Compute rotations
        R = np.zeros((N, 3, 3))
        for i, face in enumerate(F):
            for j in range(3):
                vi = V[face[j]]
                vj = V[face[(j + 1) % 3]]
                vi_new = V_new[face[j]]
                vj_new = V_new[face[(j + 1) % 3]]
                
                A = np.outer(vi - centroids[i], vj - centroids[i])
                B = np.outer(vi_new - centroids[i], vj_new - centroids[i])
                U, _, VT = np.linalg.svd(np.dot(A.T, B))
                R[face[j]] += np.dot(U, VT)

        # Step 3: Normalize rotations
        for i in range(N):
            U, _, VT = np.linalg.svd(R[i])
            R[i] = np.dot(U, VT)
        
        # Step 4: Build linear system with cotangent weights
        L = csr_matrix((N, N))
        b = np.zeros((N, 3))
        for (i, j), w in cotangent_weights.items():
            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w
            b[i] += w * (np.dot(R[i], V[j] - V[i]) + V[i])
            b[j] += w * (np.dot(R[j], V[i] - V[j]) + V[j])

        # Step 5: Apply fixed vertices constraints
        L = L.tocsr()
        for idx, pos in zip(fixed_vertices, fixed_positions):
            L[idx] = 0
            L[idx, idx] = 1
            b[idx] = pos
        
        # Step 6: Solve linear system
        V_new = spsolve(L, b)

    return V_new

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

def falloff_function(distances, influence_radii):
    """
    Computes the falloff weights based on distances and influence radii.
    
    Parameters:
    - distances: np.array of shape (num_fixed, N) containing the geodesic distances.
    - influence_radii: np.array of shape (num_fixed,) containing the influence radii for each fixed vertex.
    
    Returns:
    - falloff_weights: np.array of shape (num_fixed, N) containing the falloff weights.
    """
    return np.exp(-distances / influence_radii[:, None])

def apply_proportional_displacements(V, F, fixed_vertices, fixed_positions, influence_radii):
    """
    Applies proportional displacements to the mesh vertices based on geodesic distances and influence radii.
    
    Parameters:
    - V: np.array of shape (N, 3) containing the vertices of the mesh.
    - F: np.array of shape (M, 3) containing the faces of the mesh.
    - fixed_vertices: np.array of indices of fixed vertices.
    - fixed_positions: np.array of shape (num_fixed, 3) containing the positions of the fixed vertices.
    - influence_radii: np.array of shape (num_fixed,) containing the influence radii for each fixed vertex.
    
    Returns:
    - V_new: np.array of shape (N, 3) containing the new vertex positions after applying proportional displacements.
    """
    distances = compute_geodesic_distances(V, F, fixed_vertices)
    falloff_weights = falloff_function(distances, influence_radii)
    combined_falloff_weights = falloff_weights.sum(axis=0)
    combined_falloff_weights[combined_falloff_weights == 0] = 1  # Avoid division by zero
    
    displacements = np.zeros_like(V)
    for idx, pos in zip(fixed_vertices, fixed_positions):
        displacement = pos - V[idx]
        displacements += (falloff_weights[idx, :, None] * displacement)
    
    proportional_displacements = displacements / combined_falloff_weights[:, None]
    
    V_new = V + proportional_displacements
    return V_new

def arap_with_proportional_displacements(V, F, fixed_vertices, fixed_positions, iterations=10, influence_radii=None):
"""
    Executes the As-Rigid-As-Possible (ARAP) optimization with proportional displacements.
    
    Parameters:
    - V: np.array of shape (N, 3) containing the vertices of the mesh.
    - F: np.array of shape (M, 3) containing the faces of the mesh.
    - fixed_vertices: np.array of indices of fixed vertices.
    - fixed_positions: np.array of shape (num_fixed, 3) containing the positions of the fixed vertices.
    - iterations: int, number of iterations for the optimization process (default is 10).
    - influence_radii: np.array of shape (num_fixed,) containing the influence radii for each fixed vertex (default is None).
    
    Returns:
    - V_new: np.array of shape (N, 3) containing the new vertex positions after ARAP optimization with proportional displacements.
    """
    if influence_radii is None:
        influence_radii = np.ones(len(fixed_vertices))  # Default radius if none provided
    
    V_new = V.copy()
    
    for iter in range(iterations):
        # Run ARAP optimization
        V_new = arap(V_new, F, fixed_vertices, fixed_positions)
        
        # Apply Proportional Displacements
        V_new = apply_proportional_displacements(V_new, F, fixed_vertices, fixed_positions, influence_radii)
    
    return V_new

# Example usage
# V = np.array([...]) # Nx3 array of vertices
# F = np.array([...]) # Mx3 array of faces
# fixed_vertices = np.array([...]) # Indices of fixed vertices
# fixed_positions = np.array([...]) # Corresponding positions for fixed vertices
# influence_radii = np.array([...]) # Influence radii for each fixed vertex
# V_new = arap_with_proportional_displacements(V, F, fixed_vertices, fixed_positions, influence_radii=influence_radii)


import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra

def compute_geodesic_distances(V, F):
    """
    Computes geodesic distances between all pairs of vertices in the mesh.
    
    Parameters:
    - V: np.array of shape (N, 3) containing the vertices of the mesh.
    - F: np.array of shape (M, 3) containing the faces of the mesh.
    
    Returns:
    - distances: np.array of shape (N, N) containing the geodesic distances between each pair of vertices.
    """
    N = V.shape[0]
    graph = csr_matrix((N, N))
    for face in F:
        for i in range(3):
            i0, i1 = face[i], face[(i + 1) % 3]
            dist = np.linalg.norm(V[i0] - V[i1])
            graph[i0, i1] = dist
            graph[i1, i0] = dist
    
    distances, _ = dijkstra(graph, return_predecessors=True)
    return distances

def find_vertices_within_distance(V, F, max_distance, distance_type='geodesic'):
    """
    Finds all vertices within a given distance for each vertex in the mesh, using either geodesic or Euclidean distances.
    
    Parameters:
    - V: np.array of shape (N, 3) containing the vertices of the mesh.
    - F: np.array of shape (M, 3) containing the faces of the mesh.
    - max_distance: float, the maximum distance to consider.
    - distance_type: str, 'geodesic' or 'euclidean', the type of distance to use.
    
    Returns:
    - within_distance: list of np.arrays, where each array contains the indices of the vertices within max_distance for each vertex.
    """
    if distance_type == 'geodesic':
        distances = compute_geodesic_distances(V, F)
    elif distance_type == 'euclidean':
        distances = np.linalg.norm(V[:, np.newaxis] - V[np.newaxis, :], axis=-1)
    else:
        raise ValueError("Invalid distance_type. Choose 'geodesic' or 'euclidean'.")
    
    within_distance = []
    for i in range(len(V)):
        close_vertices = np.where(distances[i] < max_distance)[0]
        within_distance.append(close_vertices[close_vertices != i])  # Exclude the vertex itself
    
    return within_distance

def arap_with_neighbors(V, F, fixed_vertices, fixed_positions, within_distance, iterations=10):
    """
    Executes the As-Rigid-As-Possible (ARAP) optimization using neighborhoods instead of faces.
    
    Parameters:
    - V: np.array of shape (N, 3) containing the vertices of the mesh.
    - F: np.array of shape (M, 3) containing the faces of the mesh.
    - fixed_vertices: np.array of indices of fixed vertices.
    - fixed_positions: np.array of shape (num_fixed, 3) containing the positions of the fixed vertices.
    - within_distance: list of np.arrays, where each array contains the indices of the vertices within max_distance for each vertex.
    - iterations: int, number of iterations for the optimization process (default is 10).
    
    Returns:
    - V_new: np.array of shape (N, 3) containing the new vertex positions after ARAP optimization.
    """
    N = V.shape[0]
    V_new = V.copy()
    
    for iter in range(iterations):
        # Step 1: Compute centroids and rotations
        centroids = np.zeros((N, 3))
        R = np.zeros((N, 3, 3))
        for i in range(N):
            neighbors = within_distance[i]
            if len(neighbors) == 0:
                continue
            
            centroid = np.mean(V_new[neighbors], axis=0)
            centroids[i] = centroid

            A = np.zeros((3, 3))
            for j in neighbors:
                vi = V[i]
                vj = V[j]
                vi_new = V_new[i]
                vj_new = V_new[j]
                A += np.outer(vi - centroid, vj - centroid)

            U, _, VT = np.linalg.svd(A)
            R[i] = np.dot(U, VT)

        # Step 2: Build linear system using Euclidean distances as weights
        L = csr_matrix((N, N))
        b = np.zeros((N, 3))
        for i in range(N):
            neighbors = within_distance[i]
            if len(neighbors) == 0:
                continue
            
            for j in neighbors:
                w = np.linalg.norm(V[i] - V[j])  # Euclidean distance as weight
                L[i, i] += w
                L[j, j] += w
                L[i, j] -= w
                L[j, i] -= w
                b[i] += w * (np.dot(R[i], V[j] - centroids[i]) + centroids[i])
                b[j] += w * (np.dot(R[j], V[i] - centroids[i]) + centroids[i])

        # Step 3: Apply fixed vertices constraints
        L = L.tocsr()
        for idx, pos in zip(fixed_vertices, fixed_positions):
            L[idx] = 0
            L[idx, idx] = 1
            b[idx] = pos
        
        # Step 4: Solve linear system
        V_new = spsolve(L, b)

    return V_new

# Example usage
# V = np.array([...]) # Nx3 array of vertices
# F = np.array([...]) # Mx3 array of faces
# fixed_vertices = np.array([...]) # Indices of fixed vertices
# fixed_positions = np.array([...]) # Corresponding positions for fixed vertices
# max_distance = 1.0  # Maximum distance to consider for neighbors
# within_distance = find_vertices_within_distance(V, F, max_distance, distance_type='geodesic')
# V_new = arap_with_neighbors(V, F, fixed_vertices, fixed_positions, within_distance)

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
            cot_angle_2 = 0.5 / np.tan(angle)
            weights[(i1, i2)] = weights.get( (i1, i2), 0) + cot_angle_2
            weights[(i2, i1)] = weights.get( (i2, i1), 0) + cot_angle_2
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

    # Set of fixed vertex indices for fast lookup.
    fixed_vertices_set = set( fixed_vertices )

    # Also a dictionary of indices in the array by absolute vertex index for fast lookup.
    fixed_vertex_indices = {}
    for i, idx in enumerate( fixed_vertices ):
        fixed_vertex_indices[idx] = i
    
    # Initialize variables
    N         = V.shape[0]
    faces_qty = F.shape[0]
    V_new     = V.copy()
    
    for iter in range(iterations):
        
        # Compute rotations
        R = np.zeros( (N, 3, 3) )
        R_own = np.zeros( (N, 3, 3) )
        
        P_P_new = {}
        for face in F:
            for j in range(3):
                i0, i1, i2 = face[j], face[(j + 1) % 3], face[(j + 2) % 3]
                w1 = cotangent_weights[(i0, i1)]
                w2 = cotangent_weights[(i0, i2)]
                V1 = w1*(V[i0] - V[i1])
                V2 = w2*(V[i0] - V[i2])
                Pi_Pi_new = P_P_new.get( i0, ( [], [] ) )
                Pi     = Pi_Pi_new[0]
                Pi.append( V1 )
                Pi.append( V2 )

                V1_new = (V_new[i0] - V_new[i1])
                V2_new = (V_new[i0] - V_new[i2])
                Pi_new = Pi_Pi_new[1]
                Pi_new.append( V1_new )
                Pi_new.append( V2_new )

                P_P_new[i0] = ( Pi, Pi_new )

        for vert_ind in range(N):
            Pi_Pi_new = P_P_new.get( vert_ind, ( [], [] ) )
            Pi     = Pi_Pi_new[0]
            Pi_new = Pi_Pi_new[1]
            qty    = len(Pi)
            
            mPi     = np.zeros( (3, qty) )
            ind = 0
            for i, Vi in enumerate(Pi):
                mPi[:, ind] = Vi
                ind += 1

            mPi_new = np.zeros( (3, qty) )
            ind = 0
            for i, Vi in enumerate(Pi_new):
                mPi_new[:, ind] = Vi
                ind += 1

            Si = np.dot( mPi, mPi_new.T )

            # Use SVD to find the optimal rotation.
            # It transforms original points to transformed ones the best it can.
            U, _, VT = np.linalg.svd( Si )
            R[vert_ind] = np.dot(U, VT)

        # Build linear system with cotangent weights
        L = csr_matrix((N, N))
        b = np.zeros((N, 3))
        for face_ind, face in enumerate(F):
            for j in range(3):
                i0, i1, i2 = face[j], face[(j + 1) % 3], face[(j + 2) % 3]
                i0_fixed = i0 in fixed_vertices_set
                i1_fixed = i1 in fixed_vertices_set
                i2_fixed = i2 in fixed_vertices_set

                Ri0 = R[i0]
                Ri1 = R[i1]
                Ri2 = R[i2]

                w01 = cotangent_weights[ (i0, i1) ]
                w02 = cotangent_weights[ (i0, i2) ]
                if i0_fixed:
                    vert_idx = fixed_vertex_indices[i0]
                    p0       = fixed_positions[vert_idx]
                    v        = w01 * p0
                    b[i0]   -= v
                else:
                    L[i0, i0] += w01
                    L[i0, i0] += w02

                if i1_fixed:
                    vert_idx = fixed_vertex_indices[i1]
                    p1       = fixed_positions[vert_idx]
                    v        = w01 * p1
                    b[i0]   += v
                else:
                    L[i0, i1] -= w01

                if i2_fixed:
                    vert_idx = fixed_vertex_indices[i2]
                    p2       = fixed_positions[vert_idx]
                    v        = w02 * p2
                    b[i0]   += v
                else:
                    L[i0, i2] -= w02

                pi0 = V[i0]
                pi1 = V[i1]
                pi2 = V[i2]

                v1 = 0.5*w01*np.dot( (Ri0 + Ri1), (pi0 - pi1) )
                v2 = 0.5*w02*np.dot( (Ri0 + Ri2), (pi0 - pi2) )
                b[i0] += v1 + v2

        # Remove rows in both 'L' and 'b' which correspond to fixed indices.
        b = np.delete( b, fixed_vertices, axis=0 )
        # For csr_matrix it is only possible to keep, not remove. Need index inversion for that.
        # Compute indices to keep
        all_indices = np.arange(N)
        keep_indices = np.delete(all_indices, fixed_vertices)
        # Filter the rows
        L = L[keep_indices, :]
        # In L also delete columns.
        L = L[:, keep_indices]
        
        # Solve linear system
        V_some = spsolve(L, b)

        #import pdb
        #pdb.set_trace()

        # Fill in V_new matrix.
        idx = 0
        fixed_idx = 0
        for i in range(N):
            is_fixed = i in fixed_vertices_set
            if is_fixed:
                ind = fixed_vertex_indices[i]
                v = fixed_positions[ind]
                V_new[i] = v
                fixed_idx += 1

            else:
                v = V_some[idx]
                V_new[i] = v
                idx += 1

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
    if influence_radii is None:
        ret = np.zeros_like( distances )

    elif type(influence_radii) == float:
        ret = np.exp( -(distances/influence_radii)**2 )

    else:
        ret = np.exp( -(distances / influence_radii[:, None])**2 )

    return ret




def inverse_distance_weighting_multiple( falloff_func, distances, power=2, epsilon=1e-3):
    num_anchors         = distances.shape[0]
    num_points          = distances.shape[1]
    values              = falloff_func( distances )
    interpolated_values = np.zeros_like( distances )


    below_threshold_indices = np.where(distances < epsilon)
    # Create a mask for all points
    mask = np.full(distances.shape, True)
    mask[below_threshold_indices] = False
    # Get indices of points with value 1 or above by inverting the mask
    above_or_equal_threshold_indices = np.where(mask)

    for i in range(num_points):
        # Check for exact match
        if np.all( mask[:, i] ):
            # Apply inverse distance weighting
            weights = 1.0 / (distances[:, i] + epsilon)**power
            interpolated_values[:, i] = (weights * values[:, i]) / np.sum(weights)

        else:
            exact_match_index = np.argmin(distances[:, i])
            interpolated_values[exact_match_index, i] = 1.0
            print( "wull weight for vertex #", i )
            
    return interpolated_values






def apply_proportional_displacements(V, V_new, F, fixed_vertices, influence_radii):
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
    def falloff_func( distances ):
        scaled_distances = distances / influence_radii[:, None]
        ret = falloff_function( distances, influence_radii )
        return ret

    #import pdb
    #pdb.set_trace()

    distances = compute_geodesic_distances(V, F, fixed_vertices)
    falloff_weights = inverse_distance_weighting_multiple( falloff_func, distances )
    falloff_weights_2 = falloff_weights * falloff_weights
    combined_falloff_weights = falloff_weights.sum(axis=0)

    # Create a copy which is used for dividing the displacements combined.
    combined_falloff_weights_filtered = combined_falloff_weights.copy()
    combined_falloff_weights_filtered[combined_falloff_weights == 0] = 1  # Avoid division by zero
    
    displacements_arap = V_new - V
    displacements_prop = np.zeros_like(V)

    #import pdb
    #pdb.set_trace()

    N = V.shape[0]
    N_fixed = len( fixed_vertices )
    weight_idx = 0
    for idx in fixed_vertices:
        displacement = V_new[idx] - V[idx]
        displacements_prop += falloff_weights_2[weight_idx, :, None] * displacement

        weight_idx += 1
    displacements_prop    = displacements_prop / combined_falloff_weights_filtered[:, None]

    # Compute alpha for blending between proportional and arap displacements.
    alphas_prop = falloff_weights.max( axis=0 )
    alphas_arap = 1.0 - alphas_prop

    displacements = displacements_prop * alphas_prop[:, None] + displacements_arap * alphas_arap[:, None]

    V_new = V + displacements
    #V_new = V + displacements_prop
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
        influence_radii = 1.0 * np.ones(len(fixed_vertices))  # Default radius if none provided
    
    V_new = arap(V, F, fixed_vertices, fixed_positions, iterations)
    
    #V_new = V.copy()
    #V_new[fixed_vertices] = fixed_positions
    # Apply Proportional Displacements
    #V_new = apply_proportional_displacements( V, V_new, F, fixed_vertices, influence_radii)
    
    return V_new

# Example usage
# V = np.array([...]) # Nx3 array of vertices
# F = np.array([...]) # Mx3 array of faces
# fixed_vertices = np.array([...]) # Indices of fixed vertices
# fixed_positions = np.array([...]) # Corresponding positions for fixed vertices
# influence_radii = np.array([...]) # Influence radii for each fixed vertex
# V_new = arap_with_proportional_displacements(V, F, fixed_vertices, fixed_positions, influence_radii=influence_radii)


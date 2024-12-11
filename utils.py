import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra


def get_edge_lists( V, F ):
    """
    For each vertex it produces the list of all other vertices 
    it is connected with an edge.

    Returns:
    - connections, max_connections: An array of arrays of vertex indices.
    """
    verts = {}
    for face in F:
        for i in range(3):
            i0, i1, i2 = int(face[i]), int(face[(i + 1) % 3]), int(face[(i + 2) % 3])
            vert = verts.get( i0, set() )
            vert.add( i1 )
            vert.add( i2 )
            verts[i0] = vert

    # Now convert it to the array.
    connections = []
    qty = V.shape[0]
    max_connections = 0
    for i in range(qty):
        vert = verts.get( i, set() )
        # Convertes a set ot a sorted list.
        vert = sorted( vert )
        connections.append( vert )

        connections_qty = len(vert)
        if connections_qty > max_connections:
            max_connections = connections_qty


    return connections, connections_qty




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
            i0, i1, i2 = int(face[i]), int(face[(i + 1) % 3]), int(face[(i + 2) % 3])
            v0, v1, v2 = V[i0], V[i1], V[i2]
            edge1, edge2 = v1 - v0, v2 - v0
            angle = np.arccos(np.clip(np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2)), -1.0, 1.0))
            cot_angle_2 = float(0.5 / np.tan(angle))
            weights[(i1, i2)] = weights.get( (i1, i2), 0) + cot_angle_2
            weights[(i2, i1)] = weights.get( (i2, i1), 0) + cot_angle_2
    return weights

def arap(V, F, fixed_vertices, fixed_positions, iterations=2):
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
    connections, max_connections_qty = get_edge_lists(V, F)

    # Initialize variables
    N         = V.shape[0]
    faces_qty = F.shape[0]
    V_new     = V.copy()

    #import pdb
    #pdb.set_trace()

    for idx, abs_idx in enumerate(fixed_vertices):
        v = fixed_positions[idx]
        V_new[abs_idx] = v

    # Set of fixed vertex indices for fast lookup.
    fixed_vertices_set = set( fixed_vertices )

    # Also a dictionary of indices in the array by absolute vertex index for fast lookup.
    fixed_vertex_indices = {}
    for i, idx in enumerate( fixed_vertices ):
        fixed_vertex_indices[idx] = i

    # Do the same for vertex indices. Because there are fixed vertices, when constructing 
    # the optimization equation beed to skip fixed indices.
    variable_vertex_indices = {}
    idx = 0
    for i in range(N):
        is_fixed = i in fixed_vertices_set
        if is_fixed:
            continue
        variable_vertex_indices[i] = idx
        idx += 1

    variable_verts_qty = len(variable_vertex_indices)

    for iteration in range(iterations):
        
        # Compute rotations
        R = np.zeros( (N, 3, 3) )
        inv_R = np.zeros( (N, 3, 3) )
        R_to_principal = np.zeros( (N, 3, 3) )

        for abs_idx in range(N):
            # All abs vert. indices this vertex is connected to.
            vert_connections = connections[abs_idx]
            connections_qty = len(vert_connections)
            Pi = np.zeros( (connections_qty, 3) )
            Pi_new = np.zeros( (connections_qty, 3) )

            V0     = V[abs_idx]
            V0_new = V_new[abs_idx]
            
            for idx, abs_idx_other in enumerate(vert_connections):
                V1     = V[abs_idx_other]
                V1_new = V_new[abs_idx_other]
                w      = cotangent_weights[(int(abs_idx), (abs_idx_other))]
                dV     = w*(V0 - V1)
                dV_new = w*(V0_new - V1_new)

                Pi[idx] = dV
                Pi_new[idx] = dV_new

            Si = np.dot( Pi.T, Pi_new )
            # Use SVD to find the optimal rotation.
            # It transforms original points to transformed ones the best it can.
            U, _, VT = np.linalg.svd( Si )
            R_old_new = np.dot(U, VT)
            R[abs_idx] = R_old_new

            # Now compute the principal directions of original vertices set.
            #centroid = np.mean(mPi, axis=1)
            #centered_points = mPi - centroid[:, None]
            #M_cov = np.cov( centered_points )
            # Eigenvalue-eigenvector decomposition
            #eigenvalues, eigenvectors = np.linalg.eigh(M_cov)
            #sorted_indices = np.argsort(eigenvalues)
            #ei = eigenvectors[:, sorted_indices]
            # Make the smallest eigenvalue axis the last.
            #R_z = np.array( [ ei[:, 2], ei[:, 1], ei[:, 0] ] )

            # Now, instead of what they do in ARAP paper, 
            # I want to do a different thing.
            # I want to convert new points P_new to original ones.
            # And then I want to convert both to this principal axes.
            # Then, I want to scale the last axis like 1000 times so that 
            # it doesn't change the local shape.
            #inv_R[vert_ind] = R_old_new.T
            #R_to_principal[vert_ind] = R_z

        # Build linear system with cotangent weights
        L = csr_matrix( (variable_verts_qty, variable_verts_qty) )
        B = np.zeros( (variable_verts_qty, 3) )
        for abs_idx, var_idx in variable_vertex_indices.items():
            Pi     = V[abs_idx]
            Ri = R[abs_idx]
            # current vertex cannot be fixed at its index is taken from 
            # variable vertex indices.
            # However, vertices it is connected to can be fixed.
            vert_connections = connections[abs_idx]
            for abs_idx_other in vert_connections:
                Pj = V[abs_idx_other]
                Rj = R[abs_idx_other]
                # Get cotangent weight for this edge.
                w        = cotangent_weights[(int(abs_idx), int(abs_idx_other))]
                # Check if this other vertex is fixed.
                is_fixed = abs_idx_other in fixed_vertices_set
                if is_fixed:
                    # This should be the fixed vertex position.
                    vert_idx = fixed_vertex_indices[abs_idx_other]
                    Pj_new   = fixed_positions[vert_idx]
                    #Pj_new = V_new[abs_idx_other]
                    B[var_idx] += w*Pj_new

                else:
                    var_idx_other = variable_vertex_indices[abs_idx_other]
                    L[var_idx, var_idx_other] += -w

                L[var_idx, var_idx] += w
                B[var_idx]          += 0.5*w*np.dot( (Ri + Rj), (Pi - Pj) )

       
        # Solve linear system
        V_some = spsolve(L, B)

        #import pdb
        #pdb.set_trace()

        # Fill in V_new matrix.
        for abs_idx, var_idx in variable_vertex_indices.items():
            v = V_some[var_idx]
            V_new[abs_idx] = v
        
        # Fixed positions are not touched at all.
        #for idx, abs_idx in enumerate(fixed_vertex_indices):
        #    v = fixed_positions[idx]
        #    V_new[abs_idx] = v

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


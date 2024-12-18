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


def get_rigid_transform( V, fixed_vertices, fixed_positions ):
    """
    Returns rotation and translation of the best rigid transform.
    """
    start_positions = V[fixed_vertices]
    start_origin    = np.mean( start_positions, axis=0 )
    target_origin   = np.mean( fixed_positions, axis=0 )

    start_deltas    = start_positions - start_origin
    target_deltas   = fixed_positions - target_origin

    S = np.dot( start_deltas.T, target_deltas )
    U, _, VT = np.linalg.svd( S )
    inv_R = np.dot(U, VT)
    R = inv_R.T

    T = target_origin - start_origin

    return (R, T)




def get_vertex_bases( V, F ):
    """
    It first computes average normals for each vertex.
    Then, it compites a basis converting from the world ref. frame to a 
    vertex ref. frame where the normal is the first basis vector.
    """
    faces_qty = F.shape[0]
    face_normals = np.zeros( (faces_qty, 3) )
    for face_idx, face in enumerate(F):
        for i in range(3):
            i0, i1, i2 = int(face[i]), int(face[(i + 1) % 3]), int(face[(i + 2) % 3])
            v0, v1, v2 = V[i0], V[i1], V[i2]

            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)

            face_normals[face_idx] = normal

    verts_qty = V.shape[0]
    vert_normals = np.zeros( (verts_qty, 3) )

    for face_idx, face in enumerate(F):
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        face_normal = face_normals[face_idx]
        vert_normals[i0] += face_normal
        vert_normals[i1] += face_normal
        vert_normals[i2] += face_normal

    bases = np.zeros( (verts_qty, 3, 3) )

    for vert_idx in range(verts_qty):
        norm = vert_normals[vert_idx]
        abs_norm = np.linalg.norm( norm )
        if abs_norm > 1.0e-6:
            norm = norm / abs_norm
        else:
            norm = np.array( [1.0, 0.0, 0.0] )

        b = _basis_from_normal( norm )
        bases[vert_idx] = b.T

    return bases




def _basis_from_normal( n ):
    abs_n = np.abs( n )
    sorted_indices = np.argsort(-abs_n)  # Sort in descending order
    a, b, c = n[sorted_indices[0]], n[sorted_indices[1]], n[sorted_indices[2]]
    swapped = np.zeros( (3,) )
    swapped[sorted_indices[0]] = -b
    swapped[sorted_indices[1]] = a
    swapped[sorted_indices[2]] = c

    # Gram-Schmidt orthogonalization
    tangent1 = swapped - np.dot(swapped, n) * n  # Remove component along avg_normal
    tangent1 /= np.linalg.norm(tangent1)  # Normalize
    
    # Compute the third basis vector
    tangent2 = np.cross(n, tangent1)
    tangent2 /= np.linalg.norm(tangent2)  # Normalize

    basis = np.stack( [n, tangent1, tangent2], axis=0 )
    return basis





def inverse_distance_transform( V, F, fixed_vertices, fixed_positions, power=2, epsilon=1.0e-3 ):
    """
    I believe, this one implements something similar to Radial Basis Functions (RBF) based transform 
    best I understand the concept of RBF except I use geodesic distance in place of euclidean distance.
    """

    #import pdb
    #pdb.set_trace()

    R, T = get_rigid_transform( V, fixed_vertices, fixed_positions )
    # distances dims = fixed_qty by total_qty 
    distances = compute_geodesic_distances(V, F, fixed_vertices)
    # Indices where the distance is very small.
    below_threshold_indices = np.where(distances < epsilon)
    # Create a mask for all points
    mask = np.full(distances.shape, True)
    mask[below_threshold_indices] = False
 
    # Applying rigid transform, i.e. rotaton and translation.
    new_positions = np.dot( R, V.T ).T + T

    # Compute displacements for fixed positions.
    start_positions = new_positions[fixed_vertices]
    displacements   = fixed_positions - start_positions

    num_points = V.shape[0]

    for vert_idx in range(num_points):
        # Check for exact match
        if np.all( mask[:, vert_idx] ):
            # Apply inverse distance weighting
            weights = 1.0 / (distances[:, vert_idx] + epsilon)**power
            sum_weights = np.sum(weights)
            displacement_per_target = (weights[:, None] * displacements)
            displacement = np.sum( displacement_per_target, axis=0 ) / sum_weights

        else:
            exact_match_index = np.argmin(distances[:, vert_idx])
            displacement = displacements[exact_match_index]

        new_positions[vert_idx] += displacement

    return new_positions, distances



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

def arap(V, F, fixed_vertices, fixed_positions, iterations=2, V_initial=None, normal_importance=2.5):
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

    if V_initial is None:
        V_new = V.copy()
    else:
        V_new = V_initial.copy()

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
    # the optimization equation need to skip fixed indices.
    variable_vertex_indices = {}
    idx = 0
    for i in range(N):
        is_fixed = i in fixed_vertices_set
        if is_fixed:
            continue
        variable_vertex_indices[i] = idx
        idx += 1

    variable_verts_qty = len(variable_vertex_indices)
    fixed_verts_qty    = len(fixed_vertices)

    # Define start positions
    start_positions = np.zeros( (fixed_verts_qty, 3) )
    for idx, abs_idx in enumerate(fixed_vertices):
        v = V[abs_idx]
        start_positions[idx] = v

    # Convertsions from unmodified mesh to the ref. frame where vertex normal 
    # is the first coordinate.
    R_to_principal = get_vertex_bases(V, F )

    # Per-coordinate importance.
    A_scale = np.identity( 3 )
    A_scale[0,0] = normal_importance

    # Iteratively converge.
    for iteration in range(iterations):
        
        alpha = float(iteration+1)/float(iterations)
        target_positions = alpha*(fixed_positions - start_positions) + start_positions

        # Compute rotations
        R = np.zeros( (N, 3, 3) )
        inv_R = np.zeros( (N, 3, 3) )

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
            R_new_old = np.dot(U, VT)
            R_old_new = R_new_old.T
            R[abs_idx] = R_old_new
            inv_R[abs_idx] = R_new_old

        # Build linear system with cotangent weights
        L = csr_matrix( (variable_verts_qty, variable_verts_qty) )
        B = np.zeros( (variable_verts_qty, 3) )
        
        dims3 = variable_verts_qty*3
        L3 = csr_matrix( (dims3, dims3) )
        B3 = np.zeros( (dims3,) )

        for abs_idx, var_idx in variable_vertex_indices.items():
            Pi = V[abs_idx]
            Ri = R[abs_idx]

            R_new_old_i = inv_R[abs_idx]
            R_old_to_principal_i = R_to_principal[abs_idx]
            A_i = np.dot( R_old_to_principal_i, R_new_old_i )
            A_i = np.dot( A_scale, A_i )

            #A_i = R_new_old_i
            #A_i = np.identity( 3 )
            #A_i = Ri.T
            AtA_i = np.dot( A_i.T, A_i )


            # current vertex cannot be fixed at its index is taken from 
            # variable vertex indices.
            # However, vertices it is connected to can be fixed.
            vert_connections = connections[abs_idx]
            for abs_idx_other in vert_connections:
                Pj = V[abs_idx_other]
                Rj = R[abs_idx_other]

                R_new_old_j = inv_R[abs_idx_other]
                R_old_to_principal_j = R_to_principal[abs_idx_other]
                A_j = np.dot( R_old_to_principal_j, R_new_old_j )
                A_j = np.dot( A_scale, A_j )
                
                #A_j = R_new_old_j
                #A_j = np.identity( 3 )
                #A_j = Rj.T
                AtA_j = np.dot( A_j.T, A_j )

                # Get cotangent weight for this edge.
                w = cotangent_weights[(int(abs_idx), int(abs_idx_other))]

    
                A_left = w * ( AtA_i + AtA_j )
                A_right = w * ( np.dot(AtA_i, Ri) + np.dot(AtA_j, Rj) )

                # Check if this other vertex is fixed.
                is_fixed = abs_idx_other in fixed_vertices_set
                if is_fixed:
                    # This should be the fixed vertex position.
                    vert_idx = fixed_vertex_indices[abs_idx_other]
                    Pj_fixed   = fixed_positions[vert_idx]
                    #Pj_new = V_new[abs_idx_other]
                    
                    #B[var_idx] += w*Pj_fixed

                    var_idx3 = var_idx*3
                    v = w*Pj_fixed
                    v = np.dot( A_left, Pj_fixed )
                    B3[var_idx3:(var_idx3+3)] += v

                else:
                    var_idx_other = variable_vertex_indices[abs_idx_other]
                    #L[var_idx, var_idx_other] += -w

                    var_idx3 = var_idx*3
                    var_idx_other3 = var_idx_other*3
                    L3[var_idx3:(var_idx3+3), var_idx_other3:(var_idx_other3+3)] -= A_left

                #L[var_idx, var_idx] += w
                #B[var_idx]          += 0.5*w*np.dot( (Ri + Rj), (Pi - Pj) )

                var_idx3 = var_idx*3
                L3[var_idx3:(var_idx3+3), var_idx3:(var_idx3+3)] += A_left

                v = np.dot( A_right, (Pi - Pj) )
                B3[var_idx3:(var_idx3+3)] += v


       
        # Solve linear system
        #V_some = spsolve(L, B)

        V_some3 = spsolve(L3, B3)

        #import pdb
        #pdb.set_trace()

        # Fill in V_new matrix.
        for abs_idx, var_idx in variable_vertex_indices.items():
            #v = V_some[var_idx]
            #V_new[abs_idx] = v
            var_idx3 = var_idx*3
            V_new[abs_idx][0] = V_some3[var_idx3]
            V_new[abs_idx][1] = V_some3[var_idx3+1]
            V_new[abs_idx][2] = V_some3[var_idx3+2]
        
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






def apply_proportional_displacements(V_idt, V_arap, distances, influence_radii ):
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
    
    falloff_weights = inverse_distance_weighting_multiple( falloff_func, distances )
    falloff_weights_2 = falloff_weights * falloff_weights
    combined_falloff_weights = falloff_weights.sum(axis=0)

    # Create a copy which is used for dividing the displacements combined.
    combined_falloff_weights_filtered = combined_falloff_weights.copy()
    combined_falloff_weights_filtered[combined_falloff_weights == 0] = 1  # Avoid division by zero
    
    #import pdb
    #pdb.set_trace()

    # Compute alpha for blending between proportional and arap displacements.
    alphas_prop = falloff_weights.max( axis=0 )
    alphas_arap = 1.0 - alphas_prop

    V_new = V_idt * alphas_prop[:, None] + V_arap * alphas_arap[:, None]

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
    
    V_idt, distances = inverse_distance_transform( V, F, fixed_vertices, fixed_positions )
    V_arap = arap(V, F, fixed_vertices, fixed_positions, iterations, V_initial=V_idt)
    
    #V_new = V.copy()
    #V_new[fixed_vertices] = fixed_positions
    # Apply Proportional Displacements
    V_new = apply_proportional_displacements( V_idt, V_arap, distances, influence_radii )
    #V_new = V_arap
    
    return V_arap

# Example usage
# V = np.array([...]) # Nx3 array of vertices
# F = np.array([...]) # Mx3 array of faces
# fixed_vertices = np.array([...]) # Indices of fixed vertices
# fixed_positions = np.array([...]) # Corresponding positions for fixed vertices
# influence_radii = np.array([...]) # Influence radii for each fixed vertex
# V_new = arap_with_proportional_displacements(V, F, fixed_vertices, fixed_positions, influence_radii=influence_radii)


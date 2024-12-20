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



def compute_normal_importances( V, F, distances, fixed_vertices, max_importance, min_importance, influence_radii, falloff_func ):
    #import pdb
    #pdb.set_trace()
    # distances dims = [fixed_qty x total_qty]
    weights = falloff_func( distances, influence_radii )
    weights = np.max( weights, axis=0 )
    #weights = 1.0 - weights

    importances = (max_importance - min_importance) * weights + min_importance
    return importances




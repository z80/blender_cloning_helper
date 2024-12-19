import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra

from utils_distance   import *
from utils_transforms import *
from utils_geometry   import *
from utils_falloff    import *

VERY_FAR_DISTANCE = 1.0e10

def smooth_transform( V, F, fixed_data, apply_elastic=True, iterations=3, default_radius=1.0, max_influence=10.0, min_influence=1.0 ):
    import pdb
    pdb.set_trace()

    qty = len( fixed_vertices )
    fixed_positions  = np.zeros( (qty, 3) )
    fixed_vertices   = []
    distance_metrics = []
    influence_radii  = np.ones( (qty,) ) * default_radius

    for idx, vert in enumerate(fixed_data):
        index  = vert.get( "index", 0 )
        pos    = vert.get( "pos", np.zeros( (3,) ) )
        metric = vert.get( "metric", "geodesic" )
        radius = vert.get( "radius", default_radius )

        fixed_vertices.append( index )
        fixed_positions[idx] = pos
        distance_metrics[idx] = metric
        influence_radii[idx] = radius

    pdb.set_trace()

    # Compute distances from anchor points to all vertices.
    distances = compute_distances( V, F, selected_vertices, metric_types )

    # Only pick reachable vertices.
    # They are re-packed so that transform algorithms should work as usual.
    unreachable_V, unreachable_indices, \
    reachable_V, reachable_F, reachable_distances, reachable_indices \
                            = extract_reachable_vertices(Vs, Fs, fixed_vertices, fixed_positions, distances)

    # Modify reachable distances so that it doesn't contain negative numbers.
    negative_inds = np.where( reachable_distances < 0.0 )
    reachable_distances[negative_inds] = VERY_FAR_DISTANCE

    # Apply the inverse distance transform first.
    modified_reachable_V, R, T = inverse_distance_transform( reachable_V, reachable_F, reachable_fixed_vertices, reachable_fixed_positions, falloff_function )

    # Apply rigid transform to unreachable vertices.
    # That's the best we can do for them.
    modified_unreachable_V = rigid_transform( unreachable_V, R, T )

    
    # If we should run the elastic transform, do that.
    if apply_elastic:
        V_arap = arap( reachable_V, reachable_F, reachable_distances, reachable_fixed_vertices, reachable_fixed_positions, influence_radii, max_influence_min_influence, modified_reachable_V, falloff_function )
        modified_reachable_V = V_arap

    # Recombine transforms back together.
    V_new = V.copy()
    V_new[unreachable_indices] = modified_unreachable_V
    V_new[reachable_indices]   = modified_reachable_V

    return V_new











def arap_with_varible_normal_importance(V, F, fixed_vertices, fixed_positions, iterations=3, influence_radii=None):

    if influence_radii is None:
        influence_radii = 0.25 * np.ones(len(fixed_vertices))  # Default radius if none provided
    
    # Inverse distance transform is the initial approximation for the ARAP.
    V_idt, distances = inverse_distance_transform( V, F, fixed_vertices, fixed_positions )
    
    # This ARAP is with importance falloff.
    max_importance = 10.0
    min_importance = 1.0
    V_arap = arap(V, F, fixed_vertices, fixed_positions, iterations, max_importance, min_importance, influence_radii, falloff_func=falloff_function, V_initial=V_idt)

    return V_arap










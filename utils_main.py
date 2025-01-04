import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra

from utils_distance   import *
from utils_transforms import *
from utils_geometry   import *
from utils_falloff    import *

VERY_FAR_DISTANCE = 1.0e10

def smooth_transform( V, F, fixed_data, apply_gp=False, apply_elastic=True, apply_proportional_falloff=False, iterations=3, default_radius=1.0, max_influence=10.0, min_influence=1.0, normalized_gp=False ):

    qty = len( fixed_data )
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
        distance_metrics.append( metric )
        influence_radii[idx] = radius

    # Compute distances from anchor points to all vertices.
    distances = compute_distances( V, F, fixed_vertices, distance_metrics )

    # Only pick reachable vertices.
    # They are re-packed so that transform algorithms should work as usual.
    unreachable_V, unreachable_indices, \
    reachable_V, reachable_F, reachable_distances, reachable_indices \
                            = extract_reachable_vertices(V, F, fixed_vertices, fixed_positions, distances)

    # Modify reachable distances so that it doesn't contain negative numbers.
    negative_inds = np.where( reachable_distances < 0.0 )
    reachable_distances[negative_inds] = VERY_FAR_DISTANCE

    # Apply the inverse distance transform first.
    if (not apply_gp) or apply_elastic:
        modified_reachable_V, R, T = inverse_distance_transform( reachable_V, fixed_vertices, fixed_positions, reachable_distances )
        V_idt = modified_reachable_V
        V_gp  = None
    else:
        mean_radius = np.mean( influence_radii )
        modified_reachable_V, R, T = gaussian_process_transform( reachable_V, fixed_vertices, fixed_positions, reachable_distances, mean_radius, normalized=normalized_gp )
        V_idt = None
        V_gp  = modified_reachable_V

    # Apply rigid transform to unreachable vertices.
    # That's the best we can do for them.
    modified_unreachable_V = rigid_transform( unreachable_V, R, T )

    
    # If we should run the elastic transform, do that.
    if apply_elastic:
        #import pdb
        #pdb.set_trace()

        # V, F, distances, fixed_vertices, fixed_positions, iterations, max_importance, min_importance, influence_radii, falloff_func, V_initial=None
        V_arap = arap( reachable_V, reachable_F, reachable_distances, fixed_vertices, fixed_positions, iterations, max_influence, min_influence, influence_radii, falloff_function, modified_reachable_V )
        modified_reachable_V = V_arap

        if apply_proportional_falloff:
            if V_gp is None:
                V_gp, R, T = gaussian_process_transform( reachable_V, fixed_vertices, fixed_positions, reachable_distances, mean_radius, normalized=normalized_gp )
                modified_reachable_V = apply_proportional_displacements( V_gp, V_arap, reachable_distances, influence_radii )

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










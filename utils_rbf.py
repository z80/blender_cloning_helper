
import numpy as np

def RBF_transform(V, fixed_vertices, fixed_positions, distances, influence_radius, regularization=0.0001, apply_rigid_rotation=False):
    """
    Apply rigid body transformation and interpolate displacements using Gaussian Process.

    Parameters:
    V (numpy.ndarray): Nx3 array of 3D points.
    fixed_vertices (numpy.ndarray): Indices of fixed vertices (K,).
    fixed_positions (numpy.ndarray): Target positions of the fixed vertices (Kx3).
    distances (numpy.ndarray): Precomputed distances between fixed points and all points (KxN).

    Returns:
    numpy.ndarray: Updated points after applying the rigid body transform and interpolated displacements.
    """
    
    if apply_rigid_rotation:
        #Calculate rigid transform first.
        R, T = get_rigid_transform( V, fixed_vertices, fixed_positions )

        # Step 5: Apply the rigid body transform to all points
        V_transformed = np.dot(R, V.T).T + T

        # Step 6: Compute the displacements of modified points after rigid body transform
        modified_displacements = fixed_positions - V_transformed[fixed_vertices]

    else:
        V_transformed = V.copy()
        modified_displacements = fixed_positions
        R = None
        T = None

    # Step 7: Interpolate displacements using Gaussian Process
    # Use the RBF kernel (Squared Exponential)
    def rbf_kernel(distances, length_scale):
        #return np.exp( -(distances / length_scale)**2 )
        return (distances / length_scale)**2

    # Compute kernel matrices
    # K has to be symmetric. Otherwise there is no exact match in the points where the function is defined.
    K = rbf_kernel(distances[:, fixed_vertices], influence_radius)  # Kernel for fixed points
    # Add a regularization term for numerical stability
    K_regularization = regularization * np.eye(K.shape[0])
    K += K_regularization

    K_star = rbf_kernel(distances, influence_radius)                # Kernel between all points and fixed points
    decay_weights = rbf_kernel( distances, decay_radius )
    K_star *= decay_weights

    #if normalized:
    #    K_star_lengths = np.linalg.norm(K_star, axis=0)
    #    K_star = K_star / K_star_lengths[None, :]

    # Solve for weights (zero-noise GP assumes K is invertible)
    weights = np.linalg.solve(K, modified_displacements)

    # Interpolate displacements
    interpolated_displacements = np.dot(K_star.T, weights)

    # Step 8: Update all points with interpolated displacements
    V_updated = V_transformed + interpolated_displacements

    V_updated[fixed_vertices] = fixed_positions

    return V_updated, R, T





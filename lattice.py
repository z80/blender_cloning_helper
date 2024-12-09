import numpy as np
from collections import defaultdict

class Lattice:
    def __init__(self, primary_points, secondary_points, T):
        """
        Initialize the lattice.
        
        :param primary_points: Array of shape (N, 3) representing the primary points for subdivision.
        :param secondary_points: Array of shape (M, 3) representing additional points for determining lattice dimensions.
        :param T: Number of subdivisions along each axis.
        """
        self.primary_points = primary_points
        self.secondary_points = secondary_points
        self.T = T

        # Combine primary and secondary points to determine the lattice bounds
        all_points = np.vstack((primary_points, secondary_points))
        min_bounds = all_points.min(axis=0)
        max_bounds = all_points.max(axis=0)

        # Compute scale and bias for integer lattice alignment
        self.scale = (max_bounds - min_bounds) / T
        self.bias = min_bounds

        # Initialize a shared array for lattice points and a mapping for indices
        self.shared_points = []
        self.shared_indices = {}

        # Create the lattice points
        self._create_shared_lattice()

        # Precompute which primary points belong to which nodes
        self.nodes = defaultdict(list)
        self._precompute_point_membership()

    def _create_shared_lattice(self):
        """Create a shared array of lattice points."""
        for x in range(self.T + 1):
            for y in range(self.T + 1):
                for z in range(self.T + 1):
                    point = (x, y, z)
                    self.shared_indices[point] = len(self.shared_points)
                    self.shared_points.append(point)

    def _precompute_point_membership(self):
        """Precompute which primary points belong to which nodes."""
        for i, point in enumerate(self.primary_points):
            node_indices = self.point_to_node(point)
            self.nodes[tuple(node_indices)].append(i)

    def point_to_node(self, point):
        """
        Given a 3D point, compute the indices of the lattice points defining its node.
        
        :param point: A 3D point in space.
        :return: List of indices of the 8 lattice points in the shared array.
        """
        node_coords = np.floor((point - self.bias) / self.scale).astype(int)
        lower_corner = tuple(node_coords)
        upper_corner = tuple(node_coords + 1)

        # Collect the 8 lattice points defining the cube
        indices = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    corner = (lower_corner[0] + dx, lower_corner[1] + dy, lower_corner[2] + dz)
                    indices.append(self.shared_indices[corner])

        return indices

    def get_node_bounds(self, node_indices):
        """
        Get the bounds of a node from the indices of its lattice points.
        
        :param node_indices: Indices of the lattice points in the shared array.
        :return: A tuple (min_bound, max_bound) representing the bounds of the node.
        """
        corners = [np.array(self.shared_points[idx]) for idx in node_indices]
        min_bound = np.min(corners, axis=0) * self.scale + self.bias
        max_bound = np.max(corners, axis=0) * self.scale + self.bias
        return min_bound, max_bound

    def get_points_in_node(self, node_indices):
        """
        Get the primary points contained in a specific node.
        
        :param node_indices: Indices of the lattice points in the shared array.
        :return: List of primary points contained in the node.
        """
        # Extract node coordinates and convert back to integer for lookup
        lower_corner = np.array(self.shared_points[node_indices[0]])
        node_coords = tuple(lower_corner.astype(int))
        indices = self.nodes.get(node_coords, [])
        return [self.primary_points[i] for i in indices]

    def get_vertex_neighbors(self, vertex_index):
        """
        Get the neighbors of a vertex that share an edge with it.
        
        :param vertex_index: Index of the vertex in the shared array.
        :return: List of indices of neighboring vertices that share an edge.
        """
        vertex = self.shared_points[vertex_index]
        neighbors = []
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            neighbor = (vertex[0] + dx, vertex[1] + dy, vertex[2] + dz)
            if neighbor in self.shared_indices:
                neighbors.append(self.shared_indices[neighbor])
        return neighbors

    def iterate_vertices(self):
        """
        Iterate over all vertices in the lattice.
        
        :return: Generator yielding vertex indices and their neighbors.
        """
        for vertex_index, vertex_coords in enumerate(self.shared_points):
            neighbors = self.get_vertex_neighbors(vertex_index)
            yield vertex_index, vertex_coords, neighbors

# Example Usage
if __name__ == "__main__":
    # Generate primary points
    primary_points = np.random.rand(100, 3)  # 100 points in [0, 1)^3 space

    # Generate secondary points
    secondary_points = np.random.uniform(-1, 2, (50, 3))  # 50 points with wider bounds

    T = 4  # Subdivide into 4x4x4 lattice

    # Create the lattice
    lattice = Lattice(primary_points, secondary_points, T)

    # Query which node a point belongs to
    query_point = np.array([0.5, 0.5, 0.5])
    node_indices = lattice.point_to_node(query_point)
    print(f"Point {query_point} is in node defined by lattice point indices {node_indices}")

    # Get the bounds of the node
    min_bound, max_bound = lattice.get_node_bounds(node_indices)
    print(f"Node bounds: {min_bound} to {max_bound}")

    # Get points in a specific node
    points_in_node = lattice.get_points_in_node(node_indices)
    print(f"Points in node: {points_in_node}")

    # Iterate over lattice vertices and their neighbors
    print("\nLattice vertices and their neighbors:")
    for vertex_index, vertex_coords, neighbors in lattice.iterate_vertices():
        neighbor_coords = [lattice.shared_points[idx] for idx in neighbors]
        print(f"Vertex {vertex_coords} has neighbors: {neighbor_coords}")


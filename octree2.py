import numpy as np

class OctreeNode:
    def __init__(self, center, size, level, max_level, shared_points, shared_indices):
        self.center = center  # Integer center
        self.size = size  # Integer size (half-length of the cube)
        self.level = level  # Current level of subdivision
        self.max_level = max_level  # Max subdivision level
        self.shared_points = shared_points  # Shared array of integer points
        self.shared_indices = shared_indices  # Shared mapping of integer points to indices
        self.points = []  # Points contained in this node
        self.children = []  # Sub-nodes

        # Add center and vertices to shared arrays
        self.center_index = self.add_to_shared(center)
        self.vertex_indices = [self.add_to_shared(center + np.array(offset) * size)
                               for offset in self.vertex_offsets()]

    def add_to_shared(self, point):
        """Add a point to the shared array if it doesn't already exist."""
        point = tuple(point)
        if point not in self.shared_indices:
            self.shared_indices[point] = len(self.shared_points)
            self.shared_points.append(point)
        return self.shared_indices[point]

    def vertex_offsets(self):
        """Return offsets to calculate all 8 vertices of the cube."""
        return [np.array([dx, dy, dz]) for dx in [-1, 1] for dy in [-1, 1] for dz in [-1, 1]]

    def is_leaf(self):
        """Check if the node is a leaf."""
        return len(self.points) <= 1 or self.level >= self.max_level

    def subdivide(self):
        """Subdivide the node into 8 children."""
        if not self.is_leaf():
            child_size = self.size // 2
            for offset in self.vertex_offsets():
                child_center = self.center + offset * child_size
                child = OctreeNode(
                    center=child_center,
                    size=child_size,
                    level=self.level + 1,
                    max_level=self.max_level,
                    shared_points=self.shared_points,
                    shared_indices=self.shared_indices,
                )
                # Assign points to the child if they fall within its bounds
                for point in self.points:
                    if np.all(np.abs(point - child_center) <= child_size):
                        child.points.append(point)
                child.subdivide()
                self.children.append(child)

    def surface_points(self, root):
        """
        Return all points on the surface of this node, including:
        - Vertices of this node
        - Vertices of neighbors that share faces, edges, or vertices with this node
        """
        surface_points = set(self.vertex_indices)
        neighbors = self.get_neighbors(root)
        for neighbor in neighbors:
            for vertex_idx in neighbor.vertex_indices:
                vertex = np.array(self.shared_points[vertex_idx])
                if self.is_on_surface(vertex):
                    surface_points.add(vertex_idx)
        return [self.shared_points[idx] for idx in surface_points]

    def is_on_surface(self, point):
        """Check if a point lies on the surface of this node."""
        for i in range(3):  # Check each axis
            if abs(point[i] - self.center[i]) == self.size:  # On a face
                return True
        return False

    def get_neighbors(self, root):
        """Find neighbors of this node within the octree that share faces, edges, or vertices."""
        neighbors = []

        def recursive_search(node):
            if node is not self and self.intersects(node):
                neighbors.append(node)
            for child in node.children:
                recursive_search(child)

        recursive_search(root)
        return neighbors

    def intersects(self, other):
        """Check if this node intersects with another node."""
        for i in range(3):
            if abs(self.center[i] - other.center[i]) > self.size + other.size:
                return False
        return True

    def get_primary_points(self, scale, bias):
        """
        Return primary points contained in this node in floating-point format.
        If the node has no points, return an empty list.
        """
        if not self.points:
            return []
        return [(np.array(point) * scale + bias).tolist() for point in self.points]


class Octree:
    def __init__(self, primary_points, secondary_points, max_level):
        self.primary_points = primary_points
        self.secondary_points = secondary_points
        self.max_level = max_level
        self.integer_span = 2 ** max_level

        # Combine primary and secondary points to determine root bounds
        all_points = np.vstack((primary_points, secondary_points))
        min_bounds = all_points.min(axis=0)
        max_bounds = all_points.max(axis=0)

        # Compute scale and bias
        self.scale = (max_bounds - min_bounds) / (2 * self.integer_span)
        self.bias = min_bounds + self.scale * self.integer_span

        self.shared_points = []  # Shared array of integer points
        self.shared_indices = {}  # Mapping of integer points to indices in shared_points

        # Compute integer bounding box
        min_int_bounds = np.floor((all_points - self.bias) / self.scale).astype(int).min(axis=0)
        max_int_bounds = np.ceil((all_points - self.bias) / self.scale).astype(int).max(axis=0)
        root_center = (min_int_bounds + max_int_bounds) // 2
        root_size = (max_int_bounds - min_int_bounds).max() // 2

        # Create root node
        self.root = OctreeNode(
            center=root_center,
            size=root_size,
            level=0,
            max_level=max_level,
            shared_points=self.shared_points,
            shared_indices=self.shared_indices,
        )

        # Assign primary points to root and build the tree
        self.root.points = np.floor((primary_points - self.bias) / self.scale).astype(int).tolist()
        self.root.subdivide()

    def get_leaf_nodes(self):
        """Return all leaf nodes in the octree."""
        leaf_nodes = []

        def recursive_collect(node):
            if node.is_leaf():
                leaf_nodes.append(node)
            else:
                for child in node.children:
                    recursive_collect(child)

        recursive_collect(self.root)
        return leaf_nodes

    def get_surface_points(self, node):
        """Return all points on the surface of the specified node."""
        return node.surface_points(self.root)

    def convert_to_float(self, point):
        """Convert integer point to float using scale and bias."""
        return np.array(point) * self.scale + self.bias


# Example Usage
if __name__ == "__main__":
    # Generate primary points
    primary_points = np.random.rand(100, 3)  # 100 points in [0, 1)^3 space

    # Generate secondary points
    secondary_points = np.random.uniform(-1, 2, (50, 3))  # 50 points with wider bounds

    max_level = 3

    # Create the octree
    octree = Octree(primary_points, secondary_points, max_level)

    # Get all leaf nodes
    leaf_nodes = octree.get_leaf_nodes()
    print("Number of leaf nodes:", len(leaf_nodes))

    # Get primary points in each leaf node
    for i, leaf in enumerate(leaf_nodes):
        primary_points_in_leaf = leaf.get_primary_points(octree.scale, octree.bias)
        print(f"Leaf {i} contains {len(primary_points_in_leaf)} primary points.")

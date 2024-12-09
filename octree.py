import numpy as np

class OctreeNode:
    def __init__(self, center_index, size, points, V_indices=None, vertex_indices=None):
        self.center_index = center_index
        self.size = size
        self.points = points
        self.V_indices = V_indices if V_indices is not None else []
        self.vertex_indices = vertex_indices if vertex_indices is not None else []
        self.children = []

    def subdivide(self, max_level, level=0, shared_vertices=None):
        if level == max_level or len(self.points) <= 1:
            return
        
        half_size = self.size / 2
        offsets = [
            np.array([dx, dy, dz]) for dx in (-half_size, half_size)
            for dy in (-half_size, half_size) for dz in (-half_size, half_size)
        ]

        for offset in offsets:
            child_center = shared_vertices[self.center_index] + offset
            child_points = [p for p in self.points if np.all(np.abs(p - child_center) <= half_size)]
            child_V_indices = [i for i, v in zip(self.V_indices, V) if np.all(np.abs(v - child_center) <= half_size)]
            child_center_index = get_or_add_vertex(shared_vertices, child_center)
            child_vertex_indices = get_or_add_vertices(shared_vertices, child_center, half_size)
            if child_points or child_V_indices:
                child_node = OctreeNode(child_center_index, half_size, child_points, child_V_indices, child_vertex_indices)
                self.children.append(child_node)
                child_node.subdivide(max_level, level + 1, shared_vertices)

    def find_leaf_containing_point(self, point):
        if not self.children:
            return self if any(np.array_equal(point, p) for p in self.points) else None
        for child in self.children:
            if np.all(np.abs(point - shared_vertices[child.center_index]) <= child.size / 2):
                result = child.find_leaf_containing_point(point)
                if result:
                    return result
        return None

    def get_center_and_vertices(self, shared_vertices):
        center = shared_vertices[self.center_index]
        vertices = [shared_vertices[i] for i in self.vertex_indices]
        return center, vertices

    def get_center_and_vertex_indices_for_P(self, P, point_index, shared_vertices):
        leaf_node = self.find_leaf_containing_point(P[point_index])
        if leaf_node:
            return leaf_node.center_index, leaf_node.vertex_indices
        return None, None

    def get_all_leaf_nodes(self):
        if not self.children:
            return [self]
        leaf_nodes = []
        for child in self.children:
            leaf_nodes.extend(child.get_all_leaf_nodes())
        return leaf_nodes

def get_or_add_vertex(shared_vertices, vertex):
    for idx, v in enumerate(shared_vertices):
        if np.array_equal(vertex, v):
            return idx
    shared_vertices.append(vertex)
    return len(shared_vertices) - 1

def get_or_add_vertices(shared_vertices, center, size):
    half_size = size / 2
    offsets = [
        np.array([dx, dy, dz]) for dx in (-half_size, half_size)
        for dy in (-half_size, half_size) for dz in (-half_size, half_size)
    ]
    vertex_indices = []
    for offset in offsets:
        vertex = center + offset
        vertex_index = get_or_add_vertex(shared_vertices, vertex)
        vertex_indices.append(vertex_index)
    return vertex_indices

def create_octree(V, P, max_level):
    all_points = np.vstack((V, P))
    min_bounds = np.min(all_points, axis=0)
    max_bounds = np.max(all_points, axis=0)
    size = np.max(max_bounds - min_bounds)

    root_center = min_bounds + size / 2
    shared_vertices = [root_center + np.array([dx, dy, dz]) * size / 2
                       for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    root_center_index = get_or_add_vertex(shared_vertices, root_center)
    root_vertex_indices = list(range(8))

    root = OctreeNode(root_center_index, size, P, list(range(len(V))), root_vertex_indices)
    root.subdivide(max_level, shared_vertices=shared_vertices)
    return root, shared_vertices

# Example Usage
V = np.random.rand(10, 3)  # Replace with your array V
P = np.random.rand(5, 3)   # Replace with your array P
max_level = 4  # Replace with your desired maximum subdivision level

octree, shared_vertices = create_octree(V, P, max_level)

# Get all indices from V in the leaf node containing a point from P (by its index in array P)
point_index = 0  # Replace with the index of the point in P you want to find
V_indices_in_leaf = octree.get_V_indices_in_leaf_containing_point(P[point_index])

# Get center and vertex indices of the leaf node containing a point from P (by its index in array P)
center_index, vertex_indices = octree.get_center_and_vertex_indices_for_P(P, point_index, shared_vertices)

# Get all leaf nodes of the octree
leaf_nodes = octree.get_all_leaf_nodes()

print("V indices in the leaf node containing P[{}]:".format(point_index), V_indices_in_leaf)
print("Center index of the leaf node containing P[{}]:".format(point_index), center_index)
print("Vertex indices of the leaf node containing P[{}]:".format(point_index), vertex_indices)
print("Total leaf nodes in the octree:", len(leaf_nodes))
print("Shared vertices array:", shared_vertices)


import bpy
import bmesh
import mathutils
from bpy_extras.view3d_utils import location_3d_to_region_2d
from bpy_extras.view3d_utils import region_2d_to_vector_3d


import sys
import os

import numpy as np

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append( dir )

from panel_properties import *


def get_selected_mesh():
    # Get all selected objects
    selected_objects = bpy.context.selected_objects

    for obj in selected_objects:
        # Check if the object is a mesh
        if obj.type == 'MESH':
            return obj

    return None



def get_anchor_coordinates():
    """
    This is for drawing markers.
    """
    mesh = get_selected_mesh()
    if mesh is None:
        return []

    mesh_prop = mesh.data.mesh_prop
    anchors   = mesh_prop.anchors

    world_matrix = np.array( mesh.matrix_world )
    
    positions = []
    for anchor in anchors:
        pos = anchor.pos
        positions.append( pos )

    positions = np.array( positions )
    ones = np.ones( (positions.shape[0], 1) )
    homogeneous_coords = np.concatenate( (positions, ones), axis=1 )

    world_coords = homogeneous_coords @ world_matrix.T
    world_coords = world_coords[:, :3]

    # Need to turn it into the list of tuples.
    qty = world_coords.shape[0]
    list_of_coords = []
    for i in range(qty):
        list_of_coords.append( (float(world_coords[i,0]), float(world_coords[i,1]), float(world_coords[i,2]) ) )

    return list_of_coords




def get_mesh_editable( mesh ):
    mesh_prop = mesh.data.mesh_prop
    original_shape = mesh_prop.original_shape
    stored_qty = len(original_shape)

    verts_qty = len(mesh.data.vertices)

    ret = (stored_qty == verts_qty)

    return ret




def set_mesh_editable( mesh, en ):
    if en:
        store_mesh_vertices( mesh )

    else:
        mesh_prop = mesh.data.mesh_prop
        original_shape = mesh_prop.original_shape

        original_shape.clear()


def mesh_to_2d_arrays( mesh ):
    vertices = np.array([v.co for v in mesh.data.vertices])

    # Extract and virtually triangulate faces
    virtual_faces = []
    for poly in mesh.data.polygons:
        verts = poly.vertices[:]
        if len(verts) == 3:
            # If the polygon is already a triangle, use it as-is
            virtual_faces.append(verts)
        elif len(verts) > 3:
            # Triangulate the polygon
            for i in range(1, len(verts) - 1):
                virtual_faces.append((verts[0], verts[i], verts[i + 1]))

    virtual_faces = np.array(virtual_faces)

    return vertices, virtual_faces



def store_mesh_vertices( mesh ):
    mesh_prop = mesh.data.mesh_prop
    original_shape = mesh_prop.original_shape

    original_shape.clear()

    Vs, Fs = mesh_to_2d_arrays( mesh )
    for vertex in Vs:
        new_vertex = original_shape.add()
        new_vertex.pos = vertex



def add_mesh_anchor( mesh, index, pos ):
    mesh_prop = mesh.data.mesh_prop
    anchors   = mesh_prop.anchors

    for anchor in anchors:
        existing_index = anchor.index
        if index == existing_index:
            anchor.pos = pos[:]
            return

    anchor = anchors.add()
    anchor.index = index
    anchor.pos   = pos[:]




# Function to find symmetric counterpart
def find_symmetric_vertices(bm, vert, axes):
    """Find the symmetric counterpart of a vertex along the specified axis."""
    # Create a mirrored position
    mirrored_verts = [vert]
    for axis in axes:
        # Find a vertex in the mesh with the same position
        for vert in mirrored_verts:
            mirrored_pos = vert.co.copy()
            mirrored_pos[axis] = -mirrored_pos[axis]
            for other_vert in bm.verts:
                if (not other_vert.select) and (other_vert.co == mirrored_pos) and (not other_vert in mirrored_verts):
                    mirrored_verts.append( other_vert )

    return mirrored_verts


def get_mesh_update_data( mesh ):
    V, F = mesh_to_2d_arrays( mesh )

    mesh_prop = mesh.data.mesh_prop
    anchors   = mesh_prop.anchors
    
    qty = len(anchors)
    update_data = []

    for anchor in anchors:
        index  = anchor.index
        pos    = anchor.pos
        metric = anchor.metric
        radius = anchor.radius
        data = { 'index': index, 'pos': pos, 'metric': metric, 'radius': radius }

    use_gp      = (mesh_prop.step_1 == 'gaussian_proc')
    use_elastic = (mesh.prop.step_2 == 'elastic')

    iterations     = 3
    default_radius = 1.0
    max_influence = 10.0
    min_influence = 1.0

    return V, F, update_data, use_gp, use_elastic, iterations, default_radius, max_influence, min_influence



def apply_to_mesh( mesh, V_new ):
    verts = mesh.data.vertices
    verts_qty = V_new.shape[0]
    
    #import pdb
    #pdb.set_trace()
    
    mat   = mesh.matrix_world
    inv_mat = mat.inverted()
    
    for vert_ind in range(verts_qty):
        target_at = V_new[vert_ind]
        vert = verts[vert_ind]
        co = vert.co
        co.x, co.y, co.z = target_at[0], target_at[1], target_at[2]
        co = inv_mat @ co
        vert.co = co
 


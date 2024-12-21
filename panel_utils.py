
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
    



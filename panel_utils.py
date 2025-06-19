
import bpy
import bmesh
import mathutils
from bpy_extras.view3d_utils import location_3d_to_region_2d
from bpy_extras.view3d_utils import region_2d_to_vector_3d


import sys
import os
import json

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




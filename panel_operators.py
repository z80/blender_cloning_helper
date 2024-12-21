
import bpy
import bmesh
import mathutils
from bpy_extras.view3d_utils import location_3d_to_region_2d
from bpy_extras.view3d_utils import region_2d_to_vector_3d


import sys
import os

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append( dir )

from panel_properties import *
from panel_utils      import *

# Operator to add and populate the collection
class SetMeshEditableOperator(bpy.types.Operator):
    bl_idname = "object.set_mesh_editable"
    bl_label = "Make mesh editable"
    bl_description = "Add and populate a vertex collection with mesh vertex coordinates"


    @classmethod
    def poll(cls, context):
        obj = context.object
        ret = (obj is not None) and (obj.type == 'MESH') and ( not hasattr(obj, "mesh_prop") )
        return ret


    def execute(self, context):
        obj = context.object
        set_mesh_editable( True )



# Handler function to track changes
def depsgraph_update_handler(scene):
    # Check if the active object is a mesh in EDIT mode
    if bpy.context.mode != 'EDIT_MESH':
        return

    mesh = get_selected_mesh()
    if mesh is None:
        return

    editable = get_mesh_editable()
    if not editable:
        return

    # Access the bmesh for the edit mesh
    bm = bmesh.from_edit_mesh(mesh.data)
    bm.verts.ensure_lookup_table()

    # Clear the collection to avoid duplicate entries
    obj.displaced_vertices_collection.displaced_vertices.clear()

    # Check for enabled symmetries.
    axes = []
    if mesh.use_mesh_mirror_x:
        axes.append(0)
    if mesh.use_mesh_mirror_y:
        axes.append(1)
    if mesh.use_mesh_mirror_z:
        axes.append(2)

    # Record selected vertices
    for vert in bm.verts:
        if vert.select:  # Check if the vertex is selected
            verts = find_symmetric_vertices( bm, vert, axes )
            for vert_i in verts:
                index = vert_i.index
                pos   = vert_i.co[:]
                # This one either updates the existing one or adds a new one.
                add_mesh_anchor( mesh, index, pos )

    # Debug: Print the number of displaced vertices
    print(f"Displaced vertices recorded")












def register_operators():
    bpy.utils.register_class(SetMeshEditableOperator)
    # Register the depsgraph update handler
    bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_handler)



def unregister_operators():
    bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update_handler)
    bpy.utils.unregister_class(SetMeshEditableOperator)



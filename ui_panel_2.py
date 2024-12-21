
bl_info = {
    "name": "Some IGL bindings for mesh manipulation", 
    "author": "z80", 
    "version": (0, 0, 1), 
    "blender": (3, 6, 0), 
    "location": "3D Viewport > Sidebar > 1.21GW", 
    "description": "Some IGL bindings to ease mesh fitting", 
    "category": "Development", 
}



import bpy
import bmesh
import mathutils

import sys
import os

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append( dir )
    
import install_needed_packages_2

import utils_main
from panel_properties import *
from panel_operators  import *


class MESH_PT_VertexCollectionPanel(bpy.types.Panel):
    bl_label = "Mesh Vertex Collection"
    bl_idname = "VIEW3D_PT_mesh_vertex_collection"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mesh Tools"

    def draw(self, context):
        layout = self.layout

        mesh = get_selected_mesh()
        if mesh is None:
            layout.label( text="Select a mesh" )
            return
        editable = get_mesh_editable( mesh )
        if editable:
            layout.label( text="Drag vertices around in edit mode." )
            layout.label( text="Or make it not editable" )
            layout.operator("mesh.clear_mesh_editable", text="Make not editable")

        else:
            layout.operator("mesh.set_mesh_editable", text="Make editable")





def register():
    register_properties()
    register_operators()
    bpy.utils.register_class(MESH_PT_VertexCollectionPanel)


def unregister():
    bpy.utils.unregister_class(MESH_PT_VertexCollectionPanel)
    unregister_operators()
    unregister_properties()
    
    


if __name__ == "__main__":
    register()



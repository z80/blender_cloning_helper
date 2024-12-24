
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
from panel_draw       import *


class MESH_PT_MeshEditPanel(bpy.types.Panel):
    bl_label = "Mesh Edit panel"
    bl_idname = "VIEW3D_PT_mesh_edit_panel"
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

            layout.label( text="Step 1" )
            mesh_prop = mesh.data.mesh_prop
            layout.prop(mesh_prop, 'step_1', expand=True)

            layout.label( text="Step 2" )
            mesh_prop = mesh.data.mesh_prop
            layout.prop(mesh_prop, 'step_2', expand=True)

            index = get_selected_anchor( mesh )
            if index >= 0:
                anchor = mesh.data.mesh_prop.anchors[index]
                layout.label( text=f"Anchor #{index}" )
                layout.prop( anchor, 'metric', expand=True )
                layout.prop( anchor, 'radius', expand=True )

            layout.operator( "mesh.apply_transform",  text="Apply transform" )
            layout.operator( "mesh.revert_transform", text="Show original shape" )
            layout.operator( "mesh.add_anchors",      text="Make selected anchors" )
            layout.operator( "mesh.remove_anchors",   text="Remove anchors" )

        else:
            layout.operator("mesh.set_mesh_editable", text="Make editable")





def register():
    register_properties()
    register_operators()
    bpy.utils.register_class(MESH_PT_MeshEditPanel)

    register_draw()


def unregister():
    unregister_draw()

    bpy.utils.unregister_class(MESH_PT_MeshEditPanel)
    unregister_operators()
    unregister_properties()
    
    


if __name__ == "__main__":
    register()



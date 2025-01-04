
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

            mesh_prop = mesh.data.mesh_prop

            layout.label( text="Step 1" )
            layout.prop(mesh_prop, 'step_1', expand=True)

            layout.label( text="Step 2" )
            layout.prop(mesh_prop, 'step_2', expand=True)

            if mesh_prop.step_2 != 'none':
                layout.label( text="Step 3" )
                layout.prop(mesh_prop, 'step_3', expand=True)
                

            index = get_selected_anchor( mesh )
            if index >= 0:
                anchor = mesh.data.mesh_prop.anchors[index]
                layout.label( text=f"Pin #{index}" )
                layout.prop( anchor, 'metric', expand=True )
                layout.prop( anchor, 'radius', expand=True )

            layout.operator( "mesh.apply_transform",  text="Apply transform" )
            layout.operator( "mesh.revert_transform", text="Show original shape" )
            layout.operator( "mesh.add_anchors",      text="Make selected pins" )
            layout.operator( "mesh.remove_anchors",   text="Clear selected pins" )

        else:
            layout.operator("mesh.set_mesh_editable", text="Make editable")



# Panel for setting paths
class MESH_PT_ToolPathsPanel(bpy.types.Panel):
    bl_label = "Tool Settings"
    bl_idname = "MESH_PT_tool_paths"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    #bl_context = 'scene'
    bl_category = 'Mesh Tools'
    
    def draw(self, context):
        layout = self.layout
        tool_paths = context.scene.tool_paths

        layout.label(text="FFMPEG path")
        layout.prop(tool_paths, "ffmpeg_path")
        layout.label(text="Frames")
        layout.prop(tool_paths, "ffmpeg_frames")
        layout.label(text="Seconds")
        layout.prop(tool_paths, "ffmpeg_seconds")

        layout.prop(tool_paths, "colmap_path")






def register():
    register_properties()
    register_operators()
    bpy.utils.register_class(MESH_PT_MeshEditPanel)
    bpy.utils.register_class(MESH_PT_ToolPathsPanel)

    register_draw()


def unregister():
    unregister_draw()

    bpy.utils.unregister_class(MESH_PT_MeshEditPanel)
    bpy.utils.unregister_class(MESH_PT_ToolPathsPanel)
    unregister_operators()
    unregister_properties()
    
    


if __name__ == "__main__":
    register()



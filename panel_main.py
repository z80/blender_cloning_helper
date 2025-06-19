
bl_info = {
    "name": "Object cloning helper", 
    "blender": (3, 6, 0), 
    "category": "3D View", 
}

import bpy
import bmesh
import mathutils

import sys
import os

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append( dir )
    

from panel_properties import *
from panel_draw       import *

from utils_photogrammetry import *
from panel_operators_photogrammetry import *
from materials_operators import *


# Panel for setting paths
class MESH_PT_ToolPathsPanel(bpy.types.Panel):
    bl_label = "Photogrammetry"
    bl_idname = "MESH_PT_tool_paths"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    #bl_context = 'scene'
    bl_category = 'Photogrammetry'
    
    def draw(self, context):
        layout = self.layout
        tool_paths = context.scene.tool_paths

        box = layout.box()
        box.label(text="FFMPEG path")
        box.prop(tool_paths, "ffmpeg_path")

        box.prop(tool_paths, "ffmpeg_image_scale", text="Image scale %")
        box.prop(tool_paths, "ffmpeg_frames")
        box.prop(tool_paths, "ffmpeg_seconds")
        box.prop(tool_paths, "ffmpeg_start_time", text="Start time")
        box.prop(tool_paths, "ffmpeg_end_time", text="End time")

        box.operator( "wm.call_ffmpeg", text="Extract frames" )

        layout.separator()


        props = context.scene.photogrammetry_properties

        box = layout.box()
        box.prop(tool_paths, "colmap_path")
        box.operator( "wm.call_colmap", text="Extract camera poses" )
        
        layout.separator()


        box = layout.box()
        box.prop( props, 'additional_displacement', expand=True )
        box.prop( props, 'additional_rotation', expand=True )
        box.prop( props, 'additional_scale', expand=True )
        box.operator( "wm.create_ref_images", text="Create Ref Images" )

        box = layout.box()
        box.operator( "wm.assign_transform_to", text="Remember Transform To" )
        box.operator( "wm.adjust_photogrammetry_transform", text="Adjust Photogrammetry Transform" )
        box.operator( "wm.move_object_to", text="Reset Object Transform" )

        layout.separator()


        box = layout.box()
        box.label(text="Ref images")
        box.prop( props, 'show_point_cloud', expand=True )
        #box.prop( props, 'index', expand=True )
        box.prop( context.scene.photogrammetry_properties, "camera_images_items", text="Ref. Image" )
        box.operator( "wm.place_camera", text="Place camera" )
        row = box.row()
        row.operator( "wm.decrement_image_index", text="<-" )
        row.operator( "wm.increment_image_index", text="->" )

        index = props.index
        image_props = props.image_pose_properties
        qty = len(image_props)
        if (index >= 0) and (index < qty):
            image_prop = image_props[index]
            box.prop( image_prop, "user_label", text="Img. label" )

        layout.separator()


        box = layout.box()
        box.label(text="Texture paint stencil")
        box.prop( props, 'stencil_scale_adj', expand=True )
        box.operator( "wm.align_stencil", text="Align stencil" )












def register():
    register_properties()
    register_photogrammetry_props()
    register_photogrammetry()
    bpy.utils.register_class(MESH_PT_ToolPathsPanel)
    materials_register()

    register_draw()


def unregister():
    bpy.utils.unregister_class(MESH_PT_ToolPathsPanel)
    materials_unregister()
    unregister_draw()
    unregister_properties()
    unregister_photogrammetry()
    unregister_photogrammetry_props()

    
    


if __name__ == "__main__":
    register()



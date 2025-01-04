
import bpy
import os
import subprocess

from utils_photogrammetry import *

# Operator to call COLMAP
class WM_OT_CallColmap(bpy.types.Operator):
    bl_idname = "wm.call_colmap"
    bl_label = "Run COLMAP"

    def execute(self, context):
        tool_paths = context.scene.tool_paths
        colmap_path = tool_paths.colmap_path
        if not colmap_path:
            self.report({'WARNING'}, "COLMAP path is not set.")
            return {'CANCELLED'}

        call_colmap(colmap_path)

        self.report({'INFO'}, "COLMAP completed.")
        return {'FINISHED'}

# Operator to call FFmpeg
class WM_OT_CallFfmpeg(bpy.types.Operator):
    bl_idname = "wm.call_ffmpeg"
    bl_label = "Run FFmpeg"

    def execute(self, context):
        scene = context.scene.my_tool
        ffmpeg_path = scene.ffmpeg_path
        if not ffmpeg_path:
            self.report({'WARNING'}, "FFmpeg path is not set.")
            return {'CANCELLED'}

        # Example of calling FFmpeg (modify as needed)
        subprocess.run([ffmpeg_path, "-i", "input_video.mp4", "output_video.mp4"])

        self.report({'INFO'}, "FFmpeg completed.")
        return {'FINISHED'}



def register():
    bpy.utils.register_class(MyProperties)
    bpy.utils.register_class(WM_OT_CallColmap)
    bpy.utils.register_class(WM_OT_CallFfmpeg)
    bpy.utils.register_class(COLMAPPathPanel)
    bpy.utils.register_class(MainPanel)
    bpy.types.Scene.my_tool = bpy.props.PointerProperty(type=MyProperties)

def unregister():
    bpy.utils.unregister_class(MyProperties)
    bpy.utils.unregister_class(WM_OT_CallColmap)
    bpy.utils.unregister_class(WM_OT_CallFfmpeg)
    bpy.utils.unregister_class(COLMAPPathPanel)
    bpy.utils.unregister_class(MainPanel)
    del bpy.types.Scene.my_tool

if __name__ == "__main__":
    register()





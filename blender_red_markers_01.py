import bpy
import gpu
from gpu.types import GPUBatch
from mathutils import Vector

# Coordinates for the anchor points in 3D space
anchor_points = [
    Vector((1, 1, 1)),
    Vector((-1, 1, 1)),
    Vector((0, 0, 2)),
    Vector((1, -1, 1)),
    Vector((-1, -1, 1))
]

# Callback function to draw the points
def draw_callback():
    # Create a GPUVertBuf to store vertex data
    format = gpu.types.GPUVertFormat()
    pos_id = format.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
    
    vertex_buffer = gpu.types.GPUVertBuf(len=len(anchor_points), format=format)
    vertex_buffer.attr_fill(id=pos_id, data=[(point.x, point.y, point.z) for point in anchor_points])
    
    # Create the batch for drawing
    batch = gpu.types.GPUBatch(type='POINTS', buf=vertex_buffer)

    # Set the color for the points (red)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    shader.uniform_float("color", (1.0, 0.0, 0.0, 1.0))  # Red color (RGBA)

    # Set point size
    gpu.state.point_size_set(10.0)

    # Draw the batch
    batch.draw(shader)

# Operator to enable the drawing
class VIEW3D_OT_draw_red_markers(bpy.types.Operator):
    """Draw Red Markers in 3D Viewport"""
    bl_idname = "view3d.draw_red_markers"
    bl_label = "Draw Red Markers"
    bl_options = {'REGISTER', 'UNDO'}  # Ensure it appears in the search menu

    _handle = None

    @classmethod
    def poll(cls, context):
        print(f"Poll called: {context.area.type if context.area else 'None'}")
        return True  # Allow execution in any context for debugging

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            context.area.tag_redraw()
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        # Add the draw callback to Blender's draw handler
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback, (), 'WINDOW', 'POST_VIEW'
        )
        context.area.tag_redraw()
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

# Registration functions
def register():
    bpy.utils.register_class(VIEW3D_OT_draw_red_markers)
    print("Registered VIEW3D_OT_draw_red_markers")

def unregister():
    bpy.utils.unregister_class(VIEW3D_OT_draw_red_markers)

if __name__ == "__main__":
    register()


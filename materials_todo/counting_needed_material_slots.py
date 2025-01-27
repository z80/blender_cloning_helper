import bpy
import bmesh

def rectangles_containing_points(points):
    rectangles = set()
    for (x, y) in points:
        rect_x_index = int(x)
        rect_y_index = int(y)
        rectangles.add((rect_x_index, rect_y_index))
    return rectangles

def get_uv_coordinates(obj):
    uv_layer = obj.data.uv_layers.active.data
    uv_coords = [(uv.uv[0], uv.uv[1]) for uv in uv_layer]
    return uv_coords

# Ensure you're in OBJECT mode
bpy.ops.object.mode_set(mode='OBJECT')

# Get all selected objects
selected_objects = bpy.context.selected_objects

# Initialize an empty set for all unit squares
all_unit_squares = set()

# Filter for mesh objects and get UV coordinates
for obj in selected_objects:
    if obj.type == 'MESH':
        uv_coords = get_uv_coordinates(obj)
        unit_squares = rectangles_containing_points(uv_coords)
        all_unit_squares.update(unit_squares)

# Output the combined set of unit squares
print(f'Combined Unit Squares: {all_unit_squares}')


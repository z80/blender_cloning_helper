import bpy
import bmesh
import numpy as np
from PIL import Image
import os

# Create a directory to save textures
texture_dir = os.path.join(bpy.path.abspath("//"), "pbr_textures")
os.makedirs(texture_dir, exist_ok=True)

def create_pbr_texture(texture_name, resolution):
    # Create an all black texture
    data = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    img_path = os.path.join(texture_dir, f"{texture_name}.png")
    img.save(img_path)
    return img_path

def get_or_create_pbr_material(unit_square, resolution):
    material_name = f"PBR_Material_{unit_square[0]}_{unit_square[1]}"
    if material_name in bpy.data.materials:
        return bpy.data.materials[material_name]
    else:
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        # Clear existing nodes
        for node in nodes:
            nodes.remove(node)
        
        # Create Principled BSDF node
        principled_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        principled_bsdf.location = (0, 0)
        
        # Create Material Output node
        material_output = nodes.new('ShaderNodeOutputMaterial')
        material_output.location = (400, 0)
        
        # Link nodes
        links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])
        
        # Create textures and assign to material
        base_color_path = create_pbr_texture(f"{material_name}_BaseColor", resolution)
        roughness_path = create_pbr_texture(f"{material_name}_Roughness", resolution)
        normal_path = create_pbr_texture(f"{material_name}_Normal", resolution)
        
        tex_image_base = nodes.new('ShaderNodeTexImage')
        tex_image_base.image = bpy.data.images.load(base_color_path)
        tex_image_base.location = (-400, 200)
        links.new(tex_image_base.outputs['Color'], principled_bsdf.inputs['Base Color'])
        
        tex_image_roughness = nodes.new('ShaderNodeTexImage')
        tex_image_roughness.image = bpy.data.images.load(roughness_path)
        tex_image_roughness.location = (-400, 0)
        links.new(tex_image_roughness.outputs['Color'], principled_bsdf.inputs['Roughness'])
        
        tex_image_normal = nodes.new('ShaderNodeTexImage')
        tex_image_normal.image = bpy.data.images.load(normal_path)
        tex_image_normal.location = (-400, -200)
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.location = (-200, -200)
        links.new(tex_image_normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled_bsdf.inputs['Normal'])
        
        return material

def get_face_unit_square(face, uv_layer):
    avg_x = sum(uv_layer[loop_index].uv[0] for loop_index in face.loop_indices) / len(face.loop_indices)
    avg_y = sum(uv_layer[loop_index].uv[1] for loop_index in face.loop_indices) / len(face.loop_indices)
    return (int(avg_x), int(avg_y))

# Ensure you're in OBJECT mode
bpy.ops.object.mode_set(mode='OBJECT')

# Get the active object
obj = bpy.context.active_object

# Switch to Edit Mode
bpy.ops.object.mode_set(mode='EDIT')

# Get the bmesh representation
bm = bmesh.from_edit_mesh(obj.data)

# Get the UV layer
uv_layer = obj.data.uv_layers.active.data

# Specify texture resolution (e.g., 512x512)
texture_resolution = 512

# Assign PBR materials to faces based on unit squares
for face in bm.faces:
    unit_square = get_face_unit_square(face, uv_layer)
    material = get_or_create_pbr_material(unit_square, texture_resolution)
    
    # Check if the material is already in the object's material slots
    if material not in obj.data.materials:
        obj.data.materials.append(material)
    
    face.material_index = obj.data.materials.find(material.name)

# Update the mesh
bmesh.update_edit_mesh(obj.data)

# Switch back to Object Mode
bpy.ops.object.mode_set(mode='OBJECT')


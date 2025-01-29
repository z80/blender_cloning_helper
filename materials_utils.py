
import bpy
import bmesh
import numpy as np
import os

def _create_pbr_texture(texture_dir, texture_name, resolution, default_value=1.0):
    # Create a new image in Blender
    img = bpy.data.images.new(texture_name, width=resolution, height=resolution, alpha=False)
    
    # Initialize the image to black
    pixels = np.zeros((resolution, resolution, 4), dtype=np.float32)
    pixels[:, :, 3]  = 1.0  # Set alpha to 1.0
    pixels[:, :, :3] = default_value
    img.pixels = pixels.flatten()

    # Save the image
    img_path = os.path.join(texture_dir, f"{texture_name}.png")
    img.filepath_raw = img_path
    img.file_format = 'PNG'
    img.save()

    return img_path


def _convert_to_udim(x, y):
    return 1001 + x + 10 * y



def _get_or_create_pbr_material( material_name, texture_dir, unit_square, resolution, use_udim ):
    if use_udim:
        udim = _convert_to_udim( unit_square[0], unit_square[1] )
        tex_id = f"{udim}"
        material_name = f"{material_name}.{tex_id}"
    else:
        tex_id = f"{unit_square[0]}.{unit_square[1]}"
        material_name = f"{material_name}.{tex_id}"

    if material_name in bpy.data.materials:
        return bpy.data.materials[material_name]
    
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
    base_color_path = _create_pbr_texture( texture_dir, f"base_color.{tex_id}", resolution, 1.0 )
    roughness_path  = _create_pbr_texture( texture_dir, f"roughness.{tex_id}",  resolution, 0.7 )
    normal_path     = _create_pbr_texture( texture_dir, f"normal.{tex_id}",     resolution, 0.5 )
    emission_path   = _create_pbr_texture( texture_dir, f"emission.{tex_id}",   resolution, 0.0 )
    metallic_path   = _create_pbr_texture( texture_dir, f"metallic.{tex_id}",   resolution, 0.1 )
    alpha_path      = _create_pbr_texture( texture_dir, f"alpha.{tex_id}",      resolution, 1.0 )

    
    tex_image_base = nodes.new('ShaderNodeTexImage')
    tex_image_base.image = bpy.data.images.load(base_color_path)
    tex_image_base.image.colorspace_settings.name = 'sRGB'
    tex_image_base.location = (-400, 200)
    links.new(tex_image_base.outputs['Color'], principled_bsdf.inputs['Base Color'])
    
    tex_image_roughness = nodes.new('ShaderNodeTexImage')
    tex_image_roughness.image = bpy.data.images.load(roughness_path)
    tex_image_roughness.image.colorspace_settings.name = 'Non-Color'
    tex_image_roughness.location = (-400, 0)
    links.new(tex_image_roughness.outputs['Color'], principled_bsdf.inputs['Roughness'])
    
    tex_image_normal = nodes.new('ShaderNodeTexImage')
    tex_image_normal.image = bpy.data.images.load(normal_path)
    tex_image_normal.image.colorspace_settings.name = 'Non-Color'
    tex_image_normal.location = (-400, -200)
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.location = (-200, -200)
    links.new(tex_image_normal.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled_bsdf.inputs['Normal'])
    
    tex_image_emission = nodes.new('ShaderNodeTexImage')
    tex_image_emission.image = bpy.data.images.load(emission_path)
    tex_image_emission.location = (-400, -400)
    links.new(tex_image_emission.outputs['Color'], principled_bsdf.inputs['Emission Color'])

    emission_strength = nodes.new('ShaderNodeValue')
    emission_strength.outputs[0].default_value = 1.0  # Set Emission Strength to a default value (e.g., 1.0)
    links.new(emission_strength.outputs[0], principled_bsdf.inputs['Emission Strength'])

    tex_image_metallic = nodes.new('ShaderNodeTexImage')
    tex_image_metallic.image = bpy.data.images.load(metallic_path)
    tex_image_metallic.image.colorspace_settings.name = 'Non-Color'
    tex_image_metallic.location = (-400, 400)
    links.new(tex_image_metallic.outputs['Color'], principled_bsdf.inputs['Metallic'])
    
    tex_image_alpha = nodes.new('ShaderNodeTexImage')
    tex_image_alpha.image = bpy.data.images.load(alpha_path)
    tex_image_alpha.image.colorspace_settings.name = 'Non-Color'
    tex_image_alpha.location = (-400, -600)
    links.new(tex_image_alpha.outputs['Color'], principled_bsdf.inputs['Alpha'])
        
    
    return material




def _get_face_unit_square(face, uv_layer):
    #avg_x = sum(uv_layer[loop_index].uv[0] for loop_index in face.loops) / len(face.loops)
    #avg_y = sum(uv_layer[loop_index].uv[1] for loop_index in face.loops) / len(face.loops)

    loop = face.loops[0]
    avg_x = loop[uv_layer].uv.x
    avg_y = loop[uv_layer].uv.y
    return (int(avg_x), int(avg_y))



def create_udim_materials( texture_path="./textures", resolution=512, use_udim=True ):
    # Get the active object
    obj = bpy.context.active_object

    name = obj.name

    # Create a directory to save textures
    texture_dir = os.path.join(bpy.path.abspath("//"), texture_path, name )
    os.makedirs(texture_dir, exist_ok=True)    # Ensure you're in OBJECT mode

    bpy.ops.object.mode_set(mode='OBJECT')

    # Switch to Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Get the bmesh representation
    bm = bmesh.from_edit_mesh(obj.data)

    # Get the UV layer
    uv_layer = bm.loops.layers.uv.active 

    # Specify texture resolution (e.g., 512x512)
    texture_resolution = 512

    # Assign PBR materials to faces based on unit squares
    for face in bm.faces:
        unit_square = _get_face_unit_square(face, uv_layer)
        material = _get_or_create_pbr_material( name, texture_dir, unit_square, texture_resolution, use_udim )
        
        # Check if the material is already in the object's material slots
        if material.name not in obj.data.materials:
            obj.data.materials.append(material)
        
        face.material_index = obj.data.materials.find(material.name)

    # Update the mesh
    bmesh.update_edit_mesh(obj.data)

    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')






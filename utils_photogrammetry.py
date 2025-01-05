
import bpy
import os
import subprocess
import numpy as np
from mathutils import Matrix, Quaternion, Vector

def extract_frames_with_ffmpeg(video_path, ffmpeg_path, start_time, end_time, frames_qty, seconds_qty):
    # Get the directory of the current Blender file
    blender_file_dir = os.path.dirname(bpy.data.filepath)
    
    # Define the images directory path
    images_path = os.path.join(blender_file_dir, 'images')

    # Create the images directory if it doesn't exist
    os.makedirs(images_path, exist_ok=True)

    # Ensure FFmpeg path is valid
    if not os.path.isfile(ffmpeg_path):
        print(f"Invalid FFmpeg path: {ffmpeg_path}")
        return

    # Get the video duration
    result = subprocess.run([ffmpeg_path, "-i", video_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    duration_str = [x for x in result.stderr.split('\n') if "Duration" in x]
    duration_str = duration_str[0].split()[1].split(",")[0]
    hours, minutes, seconds = map(float, duration_str.split(":"))
    video_duration = hours * 3600 + minutes * 60 + seconds

    print( 'Video duration is ' + str(video_duration) + ' seconds' )

    # Adjust start time and end time
    start_time = max(0, start_time)
    if end_time < 0 or end_time > video_duration:
        end_time = video_duration

    # Calculate FPS
    fps = frames_qty / seconds_qty

    # FFmpeg command to extract frames
    ffmpeg_command = [
        ffmpeg_path,
        '-i', video_path,
        '-vf', f"fps={fps},trim=start={start_time}:end={end_time}",
        os.path.join(images_path, 'frame_%04d.png')
    ]

    print( "Done" )




def call_colmap(colmap_path):
    # Get the directory of the current Blender file
    blender_file_dir = os.path.dirname(bpy.data.filepath)
    
    # Define paths
    image_path = os.path.join(blender_file_dir, 'images')
    colmap_folder = os.path.join(blender_file_dir, 'colmap')
    photogrammetry_folder = os.path.join(blender_file_dir, 'photogrammetry')

    # Create directories if they don't exist
    os.makedirs(colmap_folder, exist_ok=True)
    os.makedirs(photogrammetry_folder, exist_ok=True)

    # Define database path
    database_path = os.path.join(colmap_folder, 'database.db')

    # Ensure COLMAP path is valid
    if not os.path.isfile(colmap_path):
        print(f"Invalid COLMAP path: {colmap_path}")
        return

    # Call COLMAP commands using the subprocess module
    print( "Extracting features" )
    subprocess.run([colmap_path, "feature_extractor", 
                    "--database_path", database_path,
                    "--image_path", image_path])

    print( "Running matcher" )
    subprocess.run([colmap_path, "matcher",
                    "--database_path", database_path])

    print( "Running mapper" )
    subprocess.run([colmap_path, "mapper",
                    "--database_path", database_path,
                    "--image_path", image_path,
                    "--output_path", colmap_folder])

    print( "Saving the results" )
    subprocess.run([colmap_path, "model_converter",
                    "--input_path", os.path.join(colmap_folder, "0"),
                    "--output_path", photogrammetry_folder,
                    "--output_type", "TXT"])

    print( "Done" )



def _read_images_file(filepath):
    image_poses = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        image_id = parts[0]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        image_name = parts[-1]
        transform_matrix = Matrix.Identity(4)
        transform_matrix.translation = Vector((tx, ty, tz))
        transform_matrix @= Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()
        image_poses[image_name] = {
            'transform': transform_matrix
        }
    return image_poses

def _read_cameras_file(filepath):
    camera_intrinsics = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        camera_id = parts[0]
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = list(map(float, parts[4:]))
        camera_intrinsics[camera_id] = {
            'model': model,
            'width': width,
            'height': height,
            'params': params
        }
    return camera_intrinsics






def populate_camera_poses():
    # Define paths
    blender_file_dir = os.path.dirname(bpy.data.filepath)
    images_file_path = os.path.join(blender_file_dir, 'photogrammetry', 'images.txt')
    cameras_file_path = os.path.join(blender_file_dir, 'photogrammetry', 'cameras.txt')

    # Read image poses and camera intrinsics
    image_poses = read_images_file(images_file_path)
    camera_intrinsics = read_cameras_file(cameras_file_path)

    # Create property groups
    bpy.context.scene.image_pose_properties.clear()
    for image_name, pose in image_poses.items():
        item = bpy.context.scene.image_pose_properties.add()
        item.image_path = os.path.join(blender_file_dir, 'images', image_name)
        item.transform = [elem for row in pose['transform'] for elem in row]
        
        # Assuming the same camera intrinsics for all images for simplicity
        camera_id = list(camera_intrinsics.keys())[0]
        intrinsics = camera_intrinsics[camera_id]
        params = intrinsics['params']
        if intrinsics['model'] == 'SIMPLE_PINHOLE':
            item.fx = item.fy = params[0]
            item.cx = params[1]
            item.cy = params[2]
        elif intrinsics['model'] == 'PINHOLE':
            item.fx = params[0]
            item.fy = params[1]
            item.cx = params[2]
            item.cy = params[3]


# Function to apply camera intrinsics and pose to the viewport
def apply_camera_settings(camera_props):
    # Get the Blender camera
    camera = bpy.data.objects['Camera']
    
    # Set camera intrinsics
    fx = camera_props.fx
    fy = camera_props.fy
    cx = camera_props.cx
    cy = camera_props.cy
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y

    camera.data.lens = fx * (camera.data.sensor_width / width)
    camera.data.shift_x = (width / 2 - cx) / width
    camera.data.shift_y = (cy - height / 2) / height

    # Set camera pose
    transform_matrix = Matrix(camera_props.transform).transposed()
    camera.matrix_world = transform_matrix

    # Set the viewport to camera view
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'


def create_image_object(camera_props, offset_distance=1.0):
    # Load the image
    image_path = camera_props.image_path
    image_name = os.path.basename(image_path)
    image = bpy.data.images.load(image_path)

    # Create a reference image object
    bpy.ops.object.add(type='EMPTY', empty_draw_type='IMAGE')
    ref_image = bpy.context.object
    ref_image.name = f"RefImage_{image_name}"
    ref_image.data = image
    ref_image.empty_image_offset = (0.5, 0.5)
    ref_image.empty_image_depth = 'DEFAULT'
    
    # Get the aspect ratio of the image
    aspect_ratio = image.size[0] / image.size[1]
    ref_image.scale = (aspect_ratio, 1, 1)

    # Set the reference image's transformation matrix
    transform_matrix = Matrix(camera_props.transform).transposed()

    # Offset the reference image slightly in front of the camera
    offset_vector = transform_matrix.to_3x3().inverted() @ Vector((0, 0, -offset_distance))
    transform_matrix.translation += offset_vector

    ref_image.matrix_world = transform_matrix


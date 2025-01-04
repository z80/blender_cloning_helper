
import bpy
import os
import subprocess

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

# Example usage
#call_colmap("path_to_colmap.bat")


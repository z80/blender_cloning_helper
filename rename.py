
import os

def rename_files(folder_path):
    # List all files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # Sort the files alphabetically
    files.sort()
    
    # Rename each file
    for index, filename in enumerate(files):
        new_name = f"{index:04}.png"
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
    
    print("Files have been renamed successfully!")


rename_files( './' )

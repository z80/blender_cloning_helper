
import os
import shutil

def contains_space_in_filename(folder_path):
    try:
        for filename in os.listdir(folder_path):
            if " " in filename:
                return True
    except FileNotFoundError:
        return False
    except PermissionError:
        return False
    return False

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


def delete_folder_contents(folder_path):
    """
    Deletes all files and subdirectories inside the given folder.

    :param folder_path: Path to the folder whose contents should be deleted.
    """
    try:
        if not os.path.exists(folder_path):
            print(f"Error: The folder '{folder_path}' does not exist.")
            return
        
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # Delete file or symbolic link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Delete directory recursively

        print(f"All contents of '{folder_path}' have been deleted.")
    except Exception as e:
        print(f"Error deleting contents of '{folder_path}': {e}")


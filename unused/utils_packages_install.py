
# Installs needed binary packages.

import subprocess
import sys
import os

import bpy

def need_packages_installed():
    #import pdb
    #pdb.set_trace()
    try:
        import numpy
        import scipy
    
    except:
        return True
    
    return False


def install_needed_packages():
    python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
    target = os.path.join(sys.prefix, 'lib', 'site-packages')
     
    subprocess.call([python_exe, '-m', 'ensurepip'])
    subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'])

    #example package to install (SciPy):
    subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'numpy', '-t', target])
    subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'scipy', '-t', target])
     
    print('DONE')



# Operator to call COLMAP
class WM_OT_MeshInstallPackages(bpy.types.Operator):
    bl_idname = "wm.mesh_install_packages"
    bl_label = "Install needed packages"

    def execute(self, context):

        install_needed_packages()

        return {'FINISHED'}



def register_install_packages():
    bpy.utils.register_class(WM_OT_MeshInstallPackages)

def unregister_install_packages():
    bpy.utils.unregister_class(WM_OT_MeshInstallPackages)

if __name__ == "__main__":
    register_install_packages()










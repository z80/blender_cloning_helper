
# Installs needed binary packages.

import subprocess
import sys
import os

def check_for_packages():
    #import pdb
    #pdb.set_trace()
    try:
        import numpy
        import scipy
        import igl
    
    except:
        return False
    
    return True


def install_needed_packages():
    python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
    target = os.path.join(sys.prefix, 'lib', 'site-packages')
     
    subprocess.call([python_exe, '-m', 'ensurepip'])
    subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'])

    #example package to install (SciPy):
    subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'numpy', '-t', target])
    subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'scipy', '-t', target])
    subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'libigl', '-t', target])
     
    print('DONE')


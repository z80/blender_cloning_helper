
import asyncio
import bpy

from utils_main import *
from panel_utils import *


async def _calculate_update( mesh, V, F, fixed_data, apply_gp, apply_elastic, iterations, default_radius, max_influence, min_influence ):
    V_new = smooth_transform( V, F, fixed_data, apply_gp, apply_elastic, iterations, default_radius, max_influence, min_influence )
    return mesh, V_new


def initiate_async_update( mesh ):
    #import pdb
    #pdb.set_trace()

    # The data needed for the update
    V, F, fixed_data, apply_gp, apply_elastic, iterations, default_radius, max_influence, min_influence = get_mesh_update_data( mesh )
    
    V_new = smooth_transform( V, F, fixed_data, apply_gp, apply_elastic, iterations, default_radius, max_influence, min_influence )

    apply_to_mesh( mesh, V_new )

    # Start the asynchronous task
    #global async_loop
    #task = loop.create_task( _calculate_update( mesh, V, F, fixed_data, apply_gp, apply_elastic, iterations, default_radius, max_influence, min_influence ) )
    #task = async_loop.create_task( _calculate_update_fake( mesh, V_new ) )
   
    #task.add_done_callback(task_done_callback)

    # Use a Blender timer to check for task completion
    #bpy.app.timers.register(lambda: _check_periodically(task), first_interval=0.5)






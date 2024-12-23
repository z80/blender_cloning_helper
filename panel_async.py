
import asyncio
import bpy

from utils_main import *
from panel_utils import *

#smooth_transform( V, F, fixed_data, apply_gp=False, apply_elastic=True, iterations=3, default_radius=1.0, max_influence=10.0, min_influence=1.0, normalized_gp=False )

async def _calculate_update( mesh, V, F, fixed_data, apply_gp, apply_elastic, iterations, default_radius, max_influence, min_influence ):
    V_new = smooth_transform( V, F, fixed_data, apply_gp, apply_elastic, iterations, default_radius, max_influence, min_influence )
    return mesh, V_new


def _check_periodically( task ):
    if task.done():
        print( "Task is complete." )
        mesh, V_new = task.result()
        # Apply to the mesh.
        apply_to_mesh( mesh, V_new )
        print( "Updated the mesh." )
        return None

    else:
        print( "Task is not done, will check again shortly..." )
        # Check again in 0.5 seconds
        return 0.5


def initiate_async_update( mesh ):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # The data needed for the update
    V, F, fixed_data, apply_gp, apply_elastic, iterations, default_radius, max_influence, min_influence = get_mesh_update_data( mesh )

    # Start the asynchronous task
    task = loop.create_task( _calculate_update( mesh, V, F, fixed_data, apply_gp, apply_elastic, iterations, default_radius, max_influence, min_influence ) )
    
    # Use a Blender timer to check for task completion
    bpy.app.timers.register(lambda: _check_periodically(task), first_interval=0.5)






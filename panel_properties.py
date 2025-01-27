
import bpy

import sys
import os
from collections import deque

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append( dir )
    

class MeshVertexProp(bpy.types.PropertyGroup):
    pos: bpy.props.FloatVectorProperty(
        name="Position",
        description="3D vector",
        size=3,  # 3 components for XYZ
        default=(0.0, 0.0, 0.0),
        subtype='XYZ'  # Display as a 3D vector
    )

class AnchorPointProp(bpy.types.PropertyGroup):
    metric: bpy.props.EnumProperty(
        name="Metric", 
        description="Distance metric", 
        items=[
            ('euclidean', 'Euclidean', 'Euclidean distance metric'), 
            ('geodesic',  'Geodesic',  'Geodesic distance metric')
        ], 
        default='euclidean'
    )

    radius: bpy.props.FloatProperty( 
        name='Radius', 
        description='Influence radius of this anchor point', 
        default=0.1, 
        min=0.0001
    )

    index: bpy.props.IntProperty( 
        name='index', 
        description='Vertex index of the anchor point', 
        default=0, 
        min=0
    )

    pos: bpy.props.FloatVectorProperty(
        name="Position",
        description="3D vector",
        size=3,  # 3 components for XYZ
        default=(0.0, 0.0, 0.0),
        subtype='XYZ'  # Display as a 3D vector
    )




class MeshProp(bpy.types.PropertyGroup):
    original_shape: bpy.props.CollectionProperty(type=MeshVertexProp)
    
    anchors: bpy.props.CollectionProperty(type=AnchorPointProp)

    step_1: bpy.props.EnumProperty(
        name="First step", 
        description="First processing step", 
        items=[
            ('inverse_dist',  'Inverse distance', 'Inverse Distance'), 
            ('gaussian_proc', 'Gaussian process',  'Gaussian Process')
        ], 
        default='inverse_dist'
    )

    # One common settings is decay radius
    decay_radius: bpy.props.FloatProperty(
        name="Decay Radius",
        description="Decay Radius Value. It defines how the whole transformation influence decays over distance",
        default=0.2
    )


    # Individual settings for the transformations.
    # Gaussian process has this one setting.
    gp_regularization: bpy.props.FloatProperty(
        name="Gaussian Process Regularization",
        description="Gaussian Process Regularization value",
        default=1.0e-6
    )

    # Inverse distance has two.
    id_epsilon: bpy.props.FloatProperty(
        name="Inverse Distance Epsilon",
        description="Inverse Distance Epsilon Value",
        default=1.0e-6
    )
    
    # Inverse distance power
    id_power: bpy.props.IntProperty( 
        name='Inverse Distance Power', 
        description='Inverse Distance Power value', 
        default=2, 
        min=1
    )

    step_2: bpy.props.EnumProperty(
        name="Second step", 
        description="Second processing step", 
        items=[
            ('none',    'None',                'No 2-d step'), 
            ('elastic', 'Elastic deformation', 'Elastic deformation transformation')
        ], 
        default='none'
    )

    step_3: bpy.props.EnumProperty(
        name="Third step", 
        description="Third processing step", 
        items=[
            ('none',    'None',                'No 3-d step'), 
            ('proportional_falloff', 'Proportional Falloff', 'Proportional Falloff between Gaussian Process and Elastic Defirmation.')
        ], 
        default='none'
    )

    update_queue: deque = deque()
    is_updating: bpy.props.BoolProperty(default=False)

    def request_update( self ):
        """
        Put request update into the queue.
        """
        pass

    def has_update_ready( self ):
        """
        Result is available.
        """
        pass

    def is_update_queued( self ):
        """
        Check if there are update requests on queue.
        """
        pass

    def initiate_update( self ):
        """
        If the update queue is not empty, 
        initiate asynchronous update.
        """
        pass








# Define a property group to store paths
class ToolProperties(bpy.types.PropertyGroup):
    colmap_path: bpy.props.StringProperty(
        name="COLMAP Path",
        description="Path to the COLMAP executable",
        default="", 
        subtype="FILE_PATH"
    )
    ffmpeg_path: bpy.props.StringProperty(
        name="FFmpeg Path",
        description="Path to the FFmpeg executable",
        default="", 
        subtype="FILE_PATH"
    )

    ffmpeg_frames: bpy.props.FloatProperty( 
        name='Frames', 
        description='Number of frames', 
        default=1.0, 
        min=0.0001
    )

    ffmpeg_seconds: bpy.props.FloatProperty( 
        name='Per Seconds', 
        description='Number of seconds', 
        default=1.0, 
        min=0.0001
    )

    ffmpeg_start_time: bpy.props.FloatProperty( 
        name='Start time in seconds', 
        description='Where in the video start extracting frames, <=0 for the beginning of the video', 
        default=-1.0, 
        min=-1.0
    )

    ffmpeg_end_time: bpy.props.FloatProperty( 
        name='End time in seconds', 
        description='Where in the video stop extracting frames, -1 for the end of the video', 
        default=-1.0, 
        min=-1.0
    )









def register_properties():
    # Register the property group
    bpy.utils.register_class(ToolProperties)
    bpy.types.Scene.tool_paths = bpy.props.PointerProperty(type=ToolProperties)

    bpy.utils.register_class(MeshVertexProp)
    bpy.utils.register_class(AnchorPointProp)
    bpy.utils.register_class(MeshProp)
    
    bpy.types.Mesh.mesh_prop = bpy.props.PointerProperty(type=MeshProp)


def unregister_properties():
    # Remove the PointerProperty from the Mesh type
    del bpy.types.Mesh.mesh_prop

    bpy.utils.unregister_class(MeshProp)
    bpy.utils.unregister_class(AnchorPointProp)
    bpy.utils.unregister_class(MeshVertexProp)

    bpy.utils.unregister_class(MyProperties)
    del bpy.types.Scene.tool_paths






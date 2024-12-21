
import bpy


import sys
import os

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
        default=1.0, 
        min=0.0001
    )

    index: bpy.props.IntProperty( 
        name='index', 
        description='Vertex index of the anchor point', 
        default=0, 
        min=0
    )

    pos: bpy.props.PointerProperty(
        name="Position",
        description="Anchor point position",
        type=MeshVertexProp
    )




class MeshProp(bpy.types.CollectionProperty):
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

    step_2: bpy.props.EnumProperty(
        name="Second step", 
        description="Second processing step", 
        items=[
            ('none',    'None',                'No 2-d step'), 
            ('elastic', 'Elastic deformation', 'Elastic deformation transformation')
        ], 
        default='none'
    )




def register_properties():
    bpy.utils.register_class(MeshVertexProp)
    bpy.utils.register_class(AnchorPointProp)
    bpy.utils.register_class(MeshProp)



def unregister_properties():
    bpy.utils.unregister_class(MeshProp)
    bpy.utils.unregister_class(AnchorPointProp)
    bpy.utils.unregister_class(MeshVertexProp)







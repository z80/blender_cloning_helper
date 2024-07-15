
bl_info = {
    "name": "Some IGL bindings for mesh manipulation", 
    "author": "z80", 
    "version": (0, 0, 1), 
    "blender": (3, 6, 0), 
    "location": "3D Viewport > Sidebar > 1.21GW", 
    "description": "Some IGL bindings to ease mesh fitting", 
    "category": "Development", 
}



import bpy
import bmesh
import mathutils
from bpy_extras.view3d_utils import location_3d_to_region_2d
from bpy_extras.view3d_utils import region_2d_to_vector_3d


import sys
import os

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append( dir )
    
import install_needed_packages


class PanelSettings(bpy.types.PropertyGroup):
    mode_enum : bpy.props.EnumProperty(
        name = "PanelMode", 
        description="Panel mode, either mesh select or addig anchors", 
        items = [("MESH_SELECT", "mesh_select", "Mesh select"), 
                 ("CREATE_ANCHORS", "crate_anchors", "Create anchors")], 
        default='MESH_SELECT'
    )

    symmetry_enum : bpy.props.EnumProperty(
        name = "SymmetryOptions",
        description = "This is a group of checkable buttons",
        items = [('NONE', "none", "Symmetry is disabled"),
                 ('X', "x", "Symmetry X"),
                 ('Y', "y", "Symmetry Y"),
                 ('Z', "z", "Symmetry Z")], 
        default='NONE'
    )

    mesh_name: bpy.props.StringProperty( 
        name="NOTHING", 
        description="The name of the selected mesh"
    )


def get_selected_mesh():
    name = bpy.context.scene.panel_settings.mesh_name
    if not ( name in bpy.data.meshes ):
        return None

    mesh = bpy.context.scene.objects[name]
    return mesh


def set_selected_mesh( mesh ):
    if mesh is None:
        name = "NOTHING"

    else:
        name = mesh.name

    bpy.context.scene.panel_settings.mesh_name = name


class VIEW3D_PT_igl_panel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "1.21 Gigawatt" # Panel top text
    bl_category = "1.21GW" # Sidebar text
    
    bl_idname = "SCENE_PT_igl_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    #bl_context = "scene"
    
    prop_live_update: bpy.props.BoolProperty(
        name="Live mesh update",
        description="If checked, mesh is updated live, else press \"Update\" button",
        default = False
        )


    def draw( self, context ):
        packages_installed = install_needed_packages.check_for_packages()
        if not packages_installed:
            self._ui_need_modules( context )
            
        else:
            #import pdb
            #pdb.set_trace()
            state = bpy.context.scene.panel_settings
            mode = state.mode_enum
            if (mode is None):
                mode = 'MESH_SELECT'
            
            if mode == 'MESH_SELECT':
                self._ui_picking_meshes( context )
                
            elif mode == 'CREATE_ANCHORS':
                self._ui_adding_ref_points( context )

 
    
    
    
    def _ui_need_modules( self, context ):
        layout = self.layout

        layout.label( text="Need python modules" )
        layout.label( text="Press the button to install" )
        layout.label( text="If it doesn't work, restart Blender" )
        layout.label( text="and try again..." )

        layout.operator("mesh.igl_install_python_modules", text="Install")
        
            
    
    def _ui_picking_meshes( self, context ):
        layout = self.layout

        layout.label( text="Select a meshes you want to stretch" )
        layout.label( text="and press the button" )
        
        # Create a simple row.
        layout.operator( "mesh.igl_pick_meshes", text="Pick a mesh" )


    def _ui_adding_ref_points( self, context ):
        layout = self.layout
    
        layout.operator( "mesh.igl_reset", text="Back to picking a mesh" )
        
        layout.label( text="Symmetry mode" )
        row = layout.row()
        panel_settings = bpy.context.scene.panel_settings
        row.prop(panel_settings, 'symmetry_enum', expand=True)
        
        layout.label( text="Click to add anchors" )
        layout.operator( "mesh.igl_create_anchor", text="Add an anchor(s)" )
        
        layout.separator()
        # Create a simple row.
        layout.label( text="Apply transform" )
        layout.operator( "mesh.igl_apply_transform", text="Apply" )
        #layout.label( text="Or return back" )
        #layout.label( text="to picking meshes" )
        #layout.operator( "mesh.igl_switch_to_editing", text="To editing" )






# Operator installing neede binary python modules.
class MESH_OT_install_python_modules( bpy.types.Operator ):
    """
    Install needed binary modules. Currently they are 
    numpy, scipy, libigl
    """
    
    bl_idname = "mesh.igl_install_python_modules"
    bl_label  = "Install needed python modules: numpy, scipy, libigl"
    
    def execute( self, context ):
        install_needed_packages.install_needed_packages()
        return {"FINISHED"}




# Operator installing needes binary python modules.
class MESH_OT_pick_selected_meshes( bpy.types.Operator ):
    """
    Picks a mesh for further editing.
    """
    
    bl_idname = "mesh.igl_pick_meshes"
    bl_label  = "Pick a mesh and edit them by adding anchors and moving them around."
    
    @classmethod
    def poll( cls, context ):
        selected_meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if len(selected_meshes) != 1:
            return False
        
        return True

    
    def execute( self, context ):
        selected_meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            
        selected_mesh = bpy.types.Mesh( selected_meshes[0] )

        state = bpy.context.scene.panel_settings
        state.mode_enum = 'CREATE_ANCHORS'
        set_selected_mesh( selected_mesh )
        return {"FINISHED"}







# Operator creating an empty axes object which is bound to 
# a vertex of an object.
class MESH_OT_create_anchor( bpy.types.Operator ):
    """
    Pick a point on a mesh by left-clicking it. An axes object should show up.
    Be aware that it only picks vertices with normals towards the camera. You 
    cannot select a vertex on a back side of a mesh.
    """
    
    bl_idname = "mesh.igl_create_anchor"
    bl_label  = "Pick all selected meshes and put them into the state."
    
    @classmethod
    def poll( cls, context ):
        # There should be a mesh in the consideration.
        state = bpy.context.scene.panel_settings
        mesh = get_selected_mesh()
        if mesh is None:
            return False
        
        return True
    
    
    def execute( self, context ):
        bpy.ops.wm.my_mouse_operator('INVOKE_DEFAULT')
        return {"FINISHED"}








class MESH_OT_apply_transform( bpy.types.Operator ):
    """
    Move anchor points around and apply the transform by clicking this button.
    """
    
    bl_idname = "mesh.igl_apply_transform"
    bl_label  = "Apply transform to the meshes selected."
    
    def execute( self, context ):
        import numpy as np
        import igl
        
        #import pdb
        #pdb.set_trace()
        
        state = bpy.context.scene.panel_settings
        mesh = get_selected_mesh()
        anchors = mesh["anchors"]
        # Before adding validate that it still exists.
        existing_anchors = []
        for anchor in anchors:
            if (anchor is not None) and (anchor.name in bpy.context.scene.objects):
                existing_anchors.append( anchor )
        anchors = existing_anchors
        mesh['anchors'] = existing_anchors
        
        Vs = mesh["verts"]
        Fs = mesh["faces"]
        #rel_vert_inds = mesh["rel_vert_inds"]
        abs_vert_inds = mesh["abs_vert_inds"]
        rel_vert_inds = {}
        for rel_ind, abs_ind in enumerate(abs_vert_inds):
            rel_vert_inds[abs_ind] = rel_ind

        # Obtain vertex indices from anchors.
        vert_inds = []
        for anchor in anchors:
            abs_vert_ind = anchor["vert_ind"]
            rel_vert_ind = rel_vert_inds[abs_vert_ind]
            vert_inds.append( rel_vert_ind )
            
        Vs, Fs = to_2d_arrays( Vs, Fs )
        
        # IGL precomputation
        arap = igl.ARAP( Vs, Fs, 3, vert_inds )
        
        target_positions = []
        for anchor in anchors:
            at = anchor.matrix_world.translation
            target_positions.append( at )
        
        target_positions = np.array( target_positions )
        
        Vs_new = arap.solve( target_positions, Vs )
        
        # Apply modified vertex coordinates to meshes.
        apply_to_mesh( mesh, Vs_new, abs_vert_inds )
        
        return {"FINISHED"}



def find_connected_vert_inds( bm, vert_ind ):
    """
    Needs bmesh obtained from a normal mesh in edit mode by doing
    bm = bmesh.new()
    bm.from_mesh(selected_mesh.data)
    """
    
    #import pdb
    #pdb.set_trace()
    
    selected_verts=set(())
    vert_inds_to_add = [vert_ind]
    added_data = True
    
    while added_data:
        added_data = False
        
        for vert_ind in vert_inds_to_add:
            if vert_ind not in selected_verts:    
                selected_verts.add( vert_ind )
                added_data = True
        
        if not added_data:
            break
        
        new_vert_inds_to_add = set(())
        for vert_ind in vert_inds_to_add:
            vertex = bm.verts[vert_ind]
    
            # Get all edges connected to the vertex
            connected_edges = vertex.link_edges
    
            connected_vertices = [edge.other_vert(vertex) for edge in connected_edges]
            connected_vert_inds = [ vert.index for vert in connected_vertices ]
            
            for vert_ind in connected_vert_inds:
                if vert_ind not in new_vert_inds_to_add:
                    new_vert_inds_to_add.add( vert_ind )
        
        vert_inds_to_add = new_vert_inds_to_add
    
    selected_verts = list(selected_verts)
    
    return selected_verts
    



def mesh_2_array( selected_mesh ):
    import numpy as np
    
    anchors = selected_mesh['anchors']
    anchor = anchors[0]
    first_vert_ind = anchor['vert_ind']
    
    bm = bmesh.new()
    bm.from_mesh(selected_mesh.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    
    #import pdb
    #pdb.set_trace()

    connected_vert_inds = find_connected_vert_inds( bm, first_vert_ind )
    
    # Create lookup tables.
    selected_ind_to_absolute_ind = []
    absolute_ind_to_selected_ind = {}
    for selected_ind, absolute_ind in enumerate(connected_vert_inds):
        selected_ind_to_absolute_ind.append( absolute_ind )
        absolute_ind_to_selected_ind[absolute_ind] = selected_ind
    
    all_verts = []
    all_faces = []
    
    # only add faces     
    for f in bm.faces:
        face = [v.index for v in f.verts]
        first_abs_ind = face[0]
        if first_abs_ind in absolute_ind_to_selected_ind:
            sel_face = []
            for i in range(3):
                vert_face_ind = face[i]
                if vert_face_ind not in absolute_ind_to_selected_ind:
                    print( "face ind ", i, " which is ", vert_face_ind, " is not in absolute_indices" )
                selected_ind = absolute_ind_to_selected_ind[face[i]]
                sel_face.append( selected_ind )
            all_faces.append( sel_face )
    bm.free()
    
    mat   = selected_mesh.matrix_world
    verts = selected_mesh.data.vertices
    verts_qty = len(selected_ind_to_absolute_ind)
    for abs_vert_ind in selected_ind_to_absolute_ind:
        vert = verts[abs_vert_ind]
        co = vert.co
        co = mat @ co
        co = [co.x, co.y, co.z]
        all_verts.append( co )
    
    
    # Convert to numpy arrays
    all_verts = np.array( all_verts )
    all_faces = np.array( all_faces )
    
    return (all_verts, all_faces, selected_ind_to_absolute_ind, absolute_ind_to_selected_ind)




def to_1d_arrays( Vs, Fs ):
    verts_qty = Vs.shape[0]
    faces_qty = Fs.shape[0]
    
    vert_coords = []
    for ind in range(verts_qty):
        x, y, z = Vs[ind, 0], Vs[ind, 1], Vs[ind, 2]
        vert_coords.append( x )
        vert_coords.append( y )
        vert_coords.append( z )
    
    face_vert_inds = []
    for ind in range(faces_qty):
        x, y, z = float(Fs[ind, 0]), float(Fs[ind, 1]), float(Fs[ind, 2])
        face_vert_inds.append( x )
        face_vert_inds.append( y )
        face_vert_inds.append( z )
    
    return (vert_coords, face_vert_inds)





def to_2d_arrays( Vs, Fs ):
    import numpy as np
    
    verts_sz = len(Vs)
    verts = []
    for ind in range( 0, verts_sz, 3 ):
        x = Vs[ind]
        y = Vs[ind+1]
        z = Vs[ind+2]
        
        verts.append( [x, y, z] )
    
    verts = np.array( verts )
    
    
    faces_sz = len(Fs)
    faces = []
    for ind in range( 0, faces_sz, 3 ):
        x = int(Fs[ind])
        y = int(Fs[ind+1])
        z = int(Fs[ind+2])
        
        faces.append( [x, y, z] )
    
    faces = np.array( faces )
    
    return (verts, faces)
        




def apply_to_mesh( mesh, Vs_new, abs_vert_inds ):
    verts = mesh.data.vertices
    verts_qty = Vs_new.shape[0]
    
    #import pdb
    #pdb.set_trace()
    
    mat   = mesh.matrix_world
    inv_mat = mat.inverted()
    
    for sel_vert_ind in range(verts_qty):
        target_at = Vs_new[sel_vert_ind]
        abs_vert_ind = abs_vert_inds[sel_vert_ind]
        vert = verts[abs_vert_ind]
        co = vert.co
        co.x, co.y, co.z = target_at[0], target_at[1], target_at[2]
        co = inv_mat @ co
        vert.co = co
        #verts[i] = vert



class MESH_OT_reset( bpy.types.Operator ):
    """
    Apply transform to selected meshes.
    """
    
    bl_idname = "mesh.igl_reset"
    bl_label  = "Reset the panel to idle state."
    
    def execute( self, context ):
        #import pdb
        #pdb.set_trace()
        
        s = bpy.context.scene.panel_settings
        s.mode_enum = 'MESH_SELECT'
        set_selected_mesh( None )
        
        return {"FINISHED"}
        





class MyMouseOperator(bpy.types.Operator):
    bl_idname = "wm.my_mouse_operator"
    bl_label = "My Mouse Operator"

    def modal(self, context, event):
        #print( "Entered SimpleMouseOperator" )
        if event.type == 'LEFTMOUSE':  # If we've clicked the left mouse button
            if event.value == 'PRESS':
                print('Left mouse button pressed')
                # Put your custom code here
            elif event.value == 'RELEASE':
                print('Left mouse button released')
                # Or put your custom code here
                self.create_anchor( context, event )
                return {'CANCELLED'}
        
        elif event.type == 'ESC':  # If we've pressed the ESC key
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    
    def create_anchor( self, context, event ):
        state = bpy.context.scene.panel_settings
        mesh = get_selected_mesh()
        if 'abs_vert_inds' in mesh:
            abs_inds = mesh['abs_vert_inds']
        
        else:
            abs_inds = None

        closest_vert_ind = self.find_closest_vertex( context, event, abs_inds )
        
        if closest_vert_ind < 0:
            return
        
        
        if 'anchors' in mesh:
            anchors = mesh['anchors']
            anchors = list(anchors)
        
        else:
            anchors = []
        
        # Filter out deleted anchors.
        existing_anchors = []
        for anchor in anchors:
            if (anchor is not None) and (anchor.name in bpy.context.scene.objects):
                existing_anchors.append( anchor )
        anchors = existing_anchors
        
        # If there are existing anchors and absolute vertex indices in the mesh, 
        # make sure that the closest selected vertex is in that list. Otherwise, 
        # different anchors are connected to isolated islands. ARAP algorithm is 
        # going to destroy the mesh in that case.
        anchors_qty = len(anchors)
        has_abs_inds = 'abs_vert_inds' in mesh
        if (anchors_qty > 0) and has_abs_inds:
            if closest_vert_ind not in abs_inds:
                return

        vert = mesh.data.vertices[closest_vert_ind]
        
        v = vert.co
        mat = mesh.matrix_world
        loc = mat @ v
        print( "Creating an anchor at ", loc )
        
        ret = bpy.ops.object.empty_add( type='PLAIN_AXES', align='WORLD', location=(loc.x, loc.y, loc.z) )
        anchor = bpy.context.active_object
        anchor.scale = (0.01, 0.01, 0.01)
        
        anchor['mirror'] = None
        anchor['symmetry'] = 'NONE'
        # Store vertex index in the anchor object.
        anchor['vert_ind'] = closest_vert_ind
        
                
        # Add the anchor created to the list of anchors ind store it in the mesh object.
        anchors.append( anchor )

        # Check symmetry.
        symmetry = bpy.context.scene.panel_settings.symmetry_enum
        if symmetry in ['X', 'Y', 'Z']:
            #import pdb
            #pdb.set_trace()

            pos = loc.copy()
            if symmetry == 'X':
                pos.x = -pos.x
            elif symmetry == 'Y':
                pos.y = -pos.y
            else:
                pos.z = -pos.z

            best_vert_ind, world_pos = self.find_closest_vertex_to_a_point( mesh, pos )
            # Make sure that we don't address one and the same vertex.
            if best_vert_ind != closest_vert_ind:
                ret = bpy.ops.object.empty_add( type='PLAIN_AXES', align='WORLD', location=(pos.x, pos.y, pos.z) )
                mirror_anchor = bpy.context.active_object
                mirror_anchor.scale = (0.01, 0.01, 0.01)
                # Store vertex index in the anchor object.
                mirror_anchor['vert_ind'] = best_vert_ind
                mirror_anchor['mirror'] = anchor
                mirror_anchor['symmetry'] = symmetry

                anchor['mirror'] = mirror_anchor
                anchor['symmetry'] = symmetry

                anchors.append( mirror_anchor )

 

        
        mesh["anchors"] = anchors
        
        # if it was the first anchor added to the mesh, store mesh data.
        if anchors_qty < 2:
            Vs, Fs, abs_vert_inds, rel_vert_inds = mesh_2_array( mesh )
            
            Vs, Fs = to_1d_arrays( Vs, Fs )
            
            #import pdb
            #pdb.set_trace()

            mesh["verts"] = Vs
            mesh["faces"] = Fs
            mesh["abs_vert_inds"] = abs_vert_inds
            # Blender cannot store a map with integer keys.
            # Will have to re'create it each time.
            #mesh["rel_vert_inds"] = rel_vert_inds


    
    
    def find_closest_vertex( self, context, event, abs_inds=None ):
        # Get the 3D view region
        region = context.region
        rv3d = context.region_data

        # Get the mouse position
        coord = mathutils.Vector( (event.mouse_region_x, event.mouse_region_y) )

        # Initialize the minimum distance and the closest vertex
        min_dist = float('inf')
        closest_vertex = None

        state  = bpy.context.scene.panel_settings

        #import pdb
        #pdb.set_trace()
        mesh = get_selected_mesh()
        
        last_best_dist = float('inf')
        
        mouse_ray = self.get_mouse_ray( context, event )
        
        best_vert_ind, best_dist = self.find_closest_vertex_in_a_mesh( region, rv3d, mesh, last_best_dist, coord, mouse_ray, abs_inds )
        
        return best_vert_ind
    
    
    
    def find_closest_vertex_in_a_mesh( self, region, rv3d, mesh, last_best_dist, mouse_at_2d, mouse_ray, abs_inds ):
        
        best_dist = last_best_dist
        best_vert_ind = -1
        
        matrix_world = mesh.matrix_world
        
        inv_matrix_world = matrix_world.to_3x3().transposed()
        mouse_ray_local = inv_matrix_world @ mouse_ray
        
        # If provided, search only within allower vertices.
        if abs_inds is not None:
            vertex_inds = abs_inds
        
        else:
            
            vertex_inds = [ vert_ind for vert_ind, vertex in enumerate( mesh.data.vertices ) ]
            
        # Iterate over all the vertices in the mesh
        for vert_ind in vertex_inds:
            vertex = mesh.data.vertices[vert_ind]            
            # Transform the vertex coordinates to 2D
            norm = vertex.normal
            
            # Check if normal is towards the camera.
            # Dot product should be negative.
            d = norm.dot( mouse_ray_local )
            if d >= 0.0:
                continue
            
            loc3d = matrix_world @ vertex.co
            loc2d = location_3d_to_region_2d( region, rv3d, loc3d )
            # It can return None, if the vertex is behind the camera.
            # And it indeed may happen because I iterate over all vertices here.
            if loc2d is None:
                continue

            # Calculate the distance to the mouse position
            dist = ( loc2d - mouse_at_2d ).length

            # If this vertex is closer, update the minimum distance and the closest vertex
            if dist < best_dist:
                best_dist     = dist
                best_vert_ind = vert_ind
        
        return best_vert_ind, best_dist
    
    
    
    def get_mouse_ray( self, context, event ):
        # Get the 3D view region
        region = context.region
        rv3d   = context.region_data

        # Get the mouse position
        coord = event.mouse_region_x, event.mouse_region_y

        # Calculate the 3D vector
        ray_vector = region_2d_to_vector_3d( region, rv3d, coord )
        
        ray_vector = ray_vector.normalized()

        return ray_vector
    
    
    
    def find_closest_vertex_to_a_point( self, mesh, point ):
    
        matrix_world = mesh.matrix_world
        
        inv_matrix_world = matrix_world.inverted()
        point_local = inv_matrix_world @ point
        
        last_best_dist = float('inf')
        last_best_ind  = -1
        world_pos = None

        vertex_inds = [ vert_ind for vert_ind, vertex in enumerate( mesh.data.vertices ) ]
            
        # Iterate over all the vertices in the mesh
        for vert_ind in vertex_inds:
            vertex = mesh.data.vertices[vert_ind]
            vertex_pos = vertex.co

            # Calculate the distance to the mouse position
            dist = ( vertex_pos - point_local ).length

            # If this vertex is closer, update the minimum distance and the closest vertex
            if dist < last_best_dist:
                last_best_dist = dist
                last_best_ind  = vert_ind
                world_pos = matrix_world @ vertex_pos
        
        return last_best_ind, world_pos




def on_transform_completed(obj, scene):
    if (obj.type != 'EMPTY'):
        return

    has_mirror = 'mirror' in obj
    if not has_mirror:
        return

    mirror = obj['mirror']
    
    if mirror is None:
        obj[symmetry] = 'NONE'
        return
    
    symmetry = obj['symmetry']

    loc = obj.location.copy()
    if symmetry == 'X':
        loc.x = -loc.x
    elif symmetry == 'Y':
        loc.y = -loc.y
    elif symmetry == 'Z':
        loc.z = -loc.z
    mirror.location = loc

    print("Transform Completed")

def on_depsgraph_update(scene, depsgraph):
    if on_depsgraph_update.operator is None:
        on_depsgraph_update.operator = bpy.context.active_operator
        return

    if on_depsgraph_update.operator == bpy.context.active_operator:
        return

    on_depsgraph_update.operator = None  # Reset now to not trigger recursion in next step in case it triggers a depsgraph update
    obj = bpy.context.active_object
    on_transform_completed(obj, scene)













def register():
    bpy.utils.register_class(PanelSettings)
    bpy.types.Scene.panel_settings = bpy.props.PointerProperty(type=PanelSettings)
    
    bpy.utils.register_class(VIEW3D_PT_igl_panel)
    bpy.utils.register_class(MESH_OT_install_python_modules)
    bpy.utils.register_class(MESH_OT_pick_selected_meshes)
    bpy.utils.register_class(MESH_OT_create_anchor)
    bpy.utils.register_class(MESH_OT_apply_transform)
    bpy.utils.register_class(MESH_OT_reset)
    
    # Make blender call on_depsgraph_update after each
    # update of Blender's internal dependency graph
    on_depsgraph_update.operator = None
    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    
    bpy.utils.register_class(MyMouseOperator)
    #bpy.ops.wm.my_mouse_operator('INVOKE_DEFAULT')


def unregister():
    bpy.utils.unregister_class(SimpleMouseOperator)
    # Make blender call on_depsgraph_update after each
    # update of Blender's internal dependency graph
    bpy.app.handlers.depsgraph_update_post.erase(on_depsgraph_update)

    bpy.utils.unregister_class(MESH_OT_install_python_modules)
    bpy.utils.unregister_class(MESH_OT_pick_selected_meshes)
    bpy.utils.unregister_class(MESH_OT_create_anchor)
    bpy.utils.unregister_class(MESH_OT_apply_transform)
    bpy.utils.unregister_class(MESH_OT_reset)
    
    bpy.utils.unregister_class(VIEW3D_PT_igl_panel)
    
    bpy.utils.unregister_class(AnchorSymmetry)
    


if __name__ == "__main__":
    register()

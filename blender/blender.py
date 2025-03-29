import bpy
import math
import mathutils
import random
import os

def clear_scene():
	"""Delete all objects, meshes, and materials."""
	bpy.ops.object.select_all(action='SELECT')
	bpy.ops.object.delete(use_global=False)
	for block in bpy.data.meshes:
		bpy.data.meshes.remove(block)
	for block in bpy.data.materials:
		bpy.data.materials.remove(block)

def setup_ground():
	"""Create a ground plane at (0,0,0) and add a passive rigid body (Convex Hull)."""
	bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
	plane = bpy.context.active_object

	# Give it a white material
	mat = bpy.data.materials.new(name="GroundMaterial")
	mat.diffuse_color = (0.2, 0.2, 0.2, 1.0)
	plane.data.materials.append(mat)
	
	# Add passive rigid body
	bpy.ops.rigidbody.object_add()
	plane.rigid_body.type = 'PASSIVE'
	plane.rigid_body.collision_shape = 'CONVEX_HULL'
	plane.rigid_body.use_margin = False
	return plane

def setup_lights():
	"""Add one main area light and eight point lights arranged in a circle."""
	lights = []
	bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
	main_light = bpy.context.active_object
	main_light.data.energy = 2500
	main_light.data.size = 5
	lights.append(main_light)
	
	num_secondary = 8
	radius = 5
	for i in range(num_secondary):
		angle = 2 * math.pi * i / num_secondary
		x = radius * math.cos(angle)
		y = radius * math.sin(angle)
		z = 5
		bpy.ops.object.light_add(type='POINT', location=(x, y, z))
		light = bpy.context.active_object
		light.data.energy = 400
		lights.append(light)
	return lights

def make_silver_material():
	"""
	Create a metallic silver material using Principled BSDF.
	Adjust Base Color, Metallic, Roughness for different looks.
	"""
	mat = bpy.data.materials.new(name="SilverMaterial")
	mat.use_nodes = True
	bsdf = mat.node_tree.nodes.get("Principled BSDF")
	if not bsdf:
		bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")

	# Light gray base color
	bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
	# Fully metallic
	bsdf.inputs["Metallic"].default_value = 1.0
	# Slightly shiny
	bsdf.inputs["Roughness"].default_value = 0.2

	return mat

def import_stl(filepath):
	"""Import STL, scale it to fit in a 5x5x5 box, then apply the scale and center the origin."""
	bpy.ops.wm.stl_import(filepath=filepath)
	obj = bpy.context.selected_objects[0]

	# Calculate bounding box dimensions
	bbox_vertices = obj.bound_box
	min_x = min(v[0] for v in bbox_vertices)
	max_x = max(v[0] for v in bbox_vertices)
	min_y = min(v[1] for v in bbox_vertices)
	max_y = max(v[1] for v in bbox_vertices)
	min_z = min(v[2] for v in bbox_vertices)
	max_z = max(v[2] for v in bbox_vertices)
	
	width = max_x - min_x
	height = max_y - min_y
	depth = max_z - min_z
	max_dimension = max(width, height, depth)
	
	# Scale so the largest dimension is 5 units
	scale_factor = 5.0 / max_dimension
	obj.scale = (scale_factor, scale_factor, scale_factor)

	# Apply the scale so physics sees the true mesh size
	bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

	# Center the origin on the geometry
	bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')

	return obj

def setup_camera():
	"""Create a camera centered above the scene looking straight down (bird's eye view)."""
	bpy.ops.object.camera_add(location=(0, 0, 15))
	cam = bpy.context.active_object
	# Point camera directly down (-Z direction)
	cam.rotation_euler = (math.radians(90), 0, 0)
	cam.data.lens = 50
	return cam

def physics_drop_chain(obj):
	"""
	Replicate the operator chain from your Scripting tab log,
	without attempting to set initial velocities directly.
	"""
	# Make sure our object is selected and active
	bpy.ops.object.select_all(action='DESELECT')
	obj.select_set(True)
	bpy.context.view_layer.objects.active = obj

	# 1) Reset the plugin
	bpy.ops.sna.pd_reset()

	# 2) Free all physics caches
	bpy.ops.ptcache.free_bake_all()

	# 3) Invert selection
	bpy.ops.object.select_all(action='INVERT')

	# 4) Deselect all
	bpy.ops.object.select_all(action='DESELECT')

	# 5) Make single user
	bpy.ops.object.make_single_user(type='SELECTED_OBJECTS', object=True, obdata=True)

	# 6) Clear parent but keep transform
	bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

	# 7) Transform apply
	bpy.ops.object.transform_apply(location=True, rotation=True, scale=True, properties=False)

	# 8) Add passive rigid body
	bpy.ops.rigidbody.objects_add(type='PASSIVE')

	# 9) Deselect all
	bpy.ops.object.select_all(action='DESELECT')

	# 10) Set origin to volume center
	bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')

	# 11) Add active rigid body
	bpy.ops.rigidbody.objects_add(type='ACTIVE')

	# Verify that a rigid body was added; if not, add it manually
	if obj.rigid_body is None:
		print("Active rigid body not found; adding manually.")
		bpy.ops.rigidbody.object_add()
		obj.rigid_body.type = 'ACTIVE'

	# 12) Finally, call the Physics Drop operator
	bpy.ops.sna.pd_drop()

	print(f"Physics Drop chain complete for object: {obj.name}")

def render_views(obj, cam, output_dir, drop_index):
	"""Render a single bird's eye view from directly above."""
	scene = bpy.context.scene
	
	# Camera is already positioned above, looking down
	scene.camera = cam
	
	# Render a single bird's eye view
	filename = f"drop_{drop_index:02d}_birds_eye.jpg"
	scene.render.filepath = os.path.join(output_dir, filename)
	bpy.ops.render.render(write_still=True)

def main():
	# Adjust paths as needed
	current_dir = os.getcwd()
	stl_filepath = os.path.join(current_dir, "training/vacuum_pump/stl/Vacuum_pump.stl")
	output_dir = os.path.join(current_dir, "training_vacuum_pump/generated")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# Use Cycles for better metallic rendering (optional but recommended)
	bpy.context.scene.render.engine = 'CYCLES'

	clear_scene()
	setup_ground()
	setup_lights()
	
	# Ensure rigid body world + gravity
	scene = bpy.context.scene
	if not scene.rigidbody_world:
		bpy.ops.rigidbody.world_add()
	scene.gravity = (0, 0, -9.81)
	
	# Import, scale, and center the object
	obj = import_stl(stl_filepath)

	# Create and assign the silver material
	silver_mat = make_silver_material()
	obj.data.materials.clear()
	obj.data.materials.append(silver_mat)

	cam = setup_camera()
	
	# Set render resolution
	scene.render.resolution_x = 256
	scene.render.resolution_y = 256
	
	num_drops = 64
	for drop in range(num_drops):
		# Randomize the object's initial orientation and slight position offset
		obj.rotation_euler = (
			random.uniform(0, 2*math.pi),
			random.uniform(0, 2*math.pi),
			random.uniform(0, 2*math.pi)
		)
		obj.location = (
			obj.location.x + random.uniform(-0.5, 0.5),
			obj.location.y + random.uniform(-0.5, 0.5),
			obj.location.z
		)
		bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
		
		# Perform the physics drop chain from your logs
		physics_drop_chain(obj)
		# Render from multiple angles
		render_views(obj, cam, output_dir, drop)
	
	print("Rendering complete!")

if __name__ == "__main__":
	main()
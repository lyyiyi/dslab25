import bpy
import math
import mathutils
import random
import os
from itertools import product

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
	bpy.ops.mesh.primitive_plane_add(size=25, location=(0, 0, 0))
	plane = bpy.context.active_object

	# Give it a dark gray material
	mat = bpy.data.materials.new(name="GroundMaterial")
	mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
	plane.data.materials.append(mat)
	
	# Add passive rigid body
	bpy.ops.rigidbody.object_add()
	plane.rigid_body.type = 'PASSIVE'
	plane.rigid_body.collision_shape = 'CONVEX_HULL'
	plane.rigid_body.use_margin = False
	# plane.hide_render = True
	return plane

def setup_lights():
	"""Add one main area light and eight point lights arranged in a circle."""
	lights = []
	bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
	main_light = bpy.context.active_object
	main_light.data.energy = 1000
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

def make_case_material():
	"""
	Create a material with the specified properties:
	Color: [0.59, 0.51, 0.43, 1.0]
	Roughness: 0.65
	Metallic: 1.0
	"""
	mat = bpy.data.materials.new(name="CaseMaterial")
	mat.use_nodes = True
	bsdf = mat.node_tree.nodes.get("Principled BSDF")
	if not bsdf:
		bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")

	bsdf.inputs["Base Color"].default_value = (0.59, 0.51, 0.43, 1.0)
	bsdf.inputs["Metallic"].default_value = 1.0
	bsdf.inputs["Roughness"].default_value = 0.65

	return mat

def make_material(texture_props):
	"""
	Create a material based on given texture properties.
	texture_props: dict with keys "color", "roughness", and "metallic".
	"""
	mat = bpy.data.materials.new(name="Material")
	mat.use_nodes = True
	bsdf = mat.node_tree.nodes.get("Principled BSDF")
	if not bsdf:
		bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
	bsdf.inputs["Base Color"].default_value = tuple(texture_props.get("color", [0.59, 0.51, 0.43, 1.0]))
	bsdf.inputs["Roughness"].default_value = texture_props.get("roughness", 0.65)
	bsdf.inputs["Metallic"].default_value = texture_props.get("metallic", 1.0)
	return mat

def import_stl(filepath):
	"""Import STL without any modification to preserve exact position from the file."""
	if not os.path.exists(filepath):
		print(f"Warning: File not found: {filepath}")
		return None
	
	bpy.ops.wm.stl_import(filepath=filepath)
	obj = bpy.context.selected_objects[0]
	return obj

def setup_camera():
	"""Create a camera at (0, 0, 15) looking straight down at the center."""
	bpy.ops.object.camera_add(location=(0, 0, 15))
	cam = bpy.context.active_object
	cam.rotation_euler = (0, 0, 0)
	cam.data.lens = 50
	return cam

def render_image(cam, output_dir, filename="render.jpg"):
	"""Render a single image from the camera's perspective."""
	scene = bpy.context.scene
	scene.camera = cam
	scene.render.filepath = os.path.join(output_dir, filename)
	bpy.ops.render.render(write_still=True)

def scatter_unused(unused_items, x_bounds=(-10, 10), y_bounds=(-10, 10), scale_factor=1.0):
	"""
	For every unused piece provided as a tuple (filepath, texture_props),
	import the STL file, set its origin and material, scale it by scale_factor,
	and assign a random location within x_bounds and y_bounds.
	Also randomize a Z-axis rotation.
	The object's name is prefixed with "UnusedScatter_" so it can be identified later.
	"""
	for item in unused_items:
		obj = import_stl(item[0])
		if obj:
			bpy.context.view_layer.objects.active = obj
			bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
			mat = make_material(item[1])
			obj.data.materials.clear()
			obj.data.materials.append(mat)
			# Random placement within specified bounds
			obj.location.x = random.uniform(x_bounds[0], x_bounds[1])
			obj.location.y = random.uniform(y_bounds[0], y_bounds[1])
			obj.location.z = 0.0
			# Random rotation around the Z-axis
			obj.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))
			# Apply the scale factor
			obj.scale = (scale_factor, scale_factor, scale_factor)
			# Name it with a tag for later removal
			obj.name = "UnusedScatter_" + os.path.basename(item[0])
			print(f"Scattered unused item: {item[0]} at {obj.location}")

def process_assembly(assembly_items, stage, perm_id=None, unused_items=None):
	# Set up GPU rendering (optional fallback inside try block in case it's not supported)
	scene = bpy.context.scene
	scene.render.engine = 'CYCLES'
	prefs = bpy.context.preferences
	try:
		prefs.addons['cycles'].preferences.compute_device_type = 'METAL'  # or 'OPTIX'/'HIP'/'METAL'
		scene.cycles.device = 'GPU'
		print("GPU rendering enabled")
	except:
		print("Could not set GPU rendering; check if your GPU is properly configured in Blender Preferences.")

	# Optional performance tweaks:
	scene.cycles.samples = 32
	scene.cycles.use_adaptive_sampling = True
	scene.cycles.max_bounces = 2
	scene.cycles.diffuse_bounces = 2
	scene.cycles.glossy_bounces = 2

	clear_scene()
	setup_ground()
	setup_lights()
	
	bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
	parent_empty = bpy.context.active_object
	parent_empty.name = "Assembly_Parent"
	
	objects = []
	for stl_item in assembly_items:
		# For items with a state flag (screws), only add if flag is 1
		if len(stl_item) == 3:
			file_path, texture_props, state_flag = stl_item
			if state_flag == 0:
				continue
		else:
			file_path, texture_props = stl_item
		obj = import_stl(file_path)
		if obj:
			bpy.context.view_layer.objects.active = obj
			bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
			mat = make_material(texture_props)
			obj.data.materials.clear()
			obj.data.materials.append(mat)
			objects.append(obj)
			obj.parent = parent_empty
	
	# Center assembly based solely on object origins
	min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
	max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
	
	for obj in objects:
		loc = obj.location
		min_x = min(min_x, loc.x)
		max_x = max(max_x, loc.x)
		min_y = min(min_y, loc.y)
		max_y = max(max_y, loc.y)
		min_z = min(min_z, loc.z)
		max_z = max(max_z, loc.z)
	
	center_x = (min_x + max_x) / 2
	center_y = (min_y + max_y) / 2
	center_z = (min_z + max_z) / 2
	
	width = max_x - min_x
	height = max_y - min_y
	depth = max_z - min_z
	max_dimension = max(width, height, depth)
	
	print(f"Assembly center: ({center_x}, {center_y}, {center_z})")
	print(f"Assembly dimensions: width={width}, height={height}, depth={depth})")
	
	parent_empty.location = (-center_x, center_y, -center_z)
	parent_empty.location[0] = -3.5
	parent_empty.location[1] = -3.5
	parent_empty.location[2] = 0.3
	
	scale_factor = 1.0
	if max_dimension > 0 and max_dimension != 5.0:
		scale_factor = 0.05
		print(f"Scaling assembly by factor: {scale_factor}")
		parent_empty.scale = (scale_factor, scale_factor, scale_factor)
	
	scene.render.resolution_x = 512
	scene.render.resolution_y = 512
	scene.render.image_settings.file_format = 'JPEG'
	# scene.render.film_transparent = True
	
	# Now for each camera render, clean up previously scattered unused items,
	# re-scatter the unused parts with new random positions, and then render.
	camera_positions = [
		(3, 3, 25), (0, 3, 25), (-3, 3, 25),
		(3, 0, 25), (0, 0, 25), (-3, 0, 25),
		(3, -3, 25), (0, -3, 25), (-3, -3, 25)
	]
	for i, pos in enumerate(camera_positions):
		bpy.ops.object.camera_add(location=pos)
		cam = bpy.context.active_object
		cam.rotation_euler = (0, 0, 0)
		cam.data.lens = 50
		if perm_id is not None:
			render_image(cam, output_dir, f"stage_{stage}_perm_{perm_id}_case_render_{i+1}.jpg")
		else:
			render_image(cam, output_dir, f"stage_{stage}_var_0_case_render_{i+1}.jpg")

	for i, pos in enumerate(camera_positions):
		# Remove previously scattered unused items.
		for obj in list(bpy.data.objects):
			if obj.name.startswith("UnusedScatter_"):
				bpy.data.objects.remove(obj, do_unlink=True)
		# Re-scatter the unused parts (if provided) for this render.
		if unused_items:
			scatter_unused(unused_items, scale_factor=scale_factor)
		
		bpy.ops.object.camera_add(location=pos)
		cam = bpy.context.active_object
		cam.rotation_euler = (0, 0, 0)
		cam.data.lens = 50
		if perm_id is not None:
			render_image(cam, output_dir, f"stage_{stage}_perm_{perm_id}_case_render_{i+1}.jpg")
		else:
			render_image(cam, output_dir, f"stage_{stage}_var_1_case_render_{i+1}.jpg")

def main():
	texture_base = {"color": [0.2, 0.2, 0.2, 1.0], "roughness": 0.5, "metallic": 0.9}
	texture_screws = {"color": [0.4, 0.4, 0.4, 1.0], "roughness": 0.65, "metallic": 1}
	texture_middle = {"color": [0.2, 0.2, 0.2, 1.0], "roughness": 0.1, "metallic": 1}
	texture_axel = {"color": [0.04, 0.035, 0.03, 1.0], "roughness": 0.5, "metallic": 0.7}
	texture_darker = {"color": [0.1, 0.1, 0.1, 1.0], "roughness": 0.65, "metallic": 0.9}
	texture_hub = {"color": [0.05, 0.05, 0.05, 1.0], "roughness": 0.6, "metallic": 1.0}
	texture_plate = {"color": [0.00, 0.00, 0.00, 1], "roughness": 1.0, "metallic": 0}
	
	stl_files = [
		[
			(os.path.join(stl_dir, "1_base.stl"), texture_base),
			(os.path.join(stl_dir, "1_base_deckel.stl"), texture_base),
			(os.path.join(stl_dir, "1_base_deckel_screws.stl"), texture_screws)
		],
		[
			(os.path.join(stl_dir, "2_axel.stl"), texture_axel)
		],
		[
			(os.path.join(stl_dir, "3_middle_part.stl"), texture_base),
			(os.path.join(stl_dir, "3_middle.stl"), texture_base)
		], 
		[
			(os.path.join(stl_dir, "4_diamond.stl"), texture_darker)
		],
		[
			(os.path.join(stl_dir, "5_new.stl"), texture_hub)
			# (os.path.join(stl_dir, "5_hub.stl"), texture_hub)
		], 
		[
			(os.path.join(stl_dir, "6_screws_01.stl"), texture_screws, True),
			(os.path.join(stl_dir, "6_screws_02.stl"), texture_screws, True),
			(os.path.join(stl_dir, "6_screws_03.stl"), texture_screws, True)
		],
		[
			(os.path.join(stl_dir, "7_plate.stl"), texture_plate)
		],
		[
			(os.path.join(stl_dir, "8_screws_01.stl"), texture_screws, True),
			(os.path.join(stl_dir, "8_screws_02.stl"), texture_screws, True),
			(os.path.join(stl_dir, "8_screws_03.stl"), texture_screws, True),
			(os.path.join(stl_dir, "8_screws_04.stl"), texture_screws, True),
			(os.path.join(stl_dir, "8_screws_05.stl"), texture_screws, True)
		]
	]
	
	curr_stls = []
	max_scatter = 3  # Maximum number of unused pieces to scatter on the ground.
	for stage, stl_group in enumerate(stl_files):
		# Gather unused parts from future stages for scattering.
		unused = []
		for future_group in stl_files[stage+1:]:
			for item in future_group:
				# Only take the filepath and texture_props
				unused.append(item[:2])
		# scatter_selection = random.sample(unused, min(len(unused), max_scatter)) if unused else None
		scatter_selection = unused

		# Check if the group is permutable (the screws group)
		if stl_group and len(stl_group[0]) == 3 and stl_group[0][2] is True and False:
			combos = list(product([0, 1], repeat=len(stl_group)))
			for c_index, combo in enumerate(combos):
				# Skip the case where no screw is added
				if sum(combo) == 0:
					continue
				new_group = []
				for idx, screw_item in enumerate(stl_group):
					# Only include the screw if the flag in this combo is 1
					new_group.append((screw_item[0], screw_item[1], combo[idx]))
				temp_stls = curr_stls + new_group
				process_assembly(temp_stls, stage, c_index, unused_items=scatter_selection)
			# For further stages, add the default state (all screws included, flag = 1)
			curr_stls += [(item[0], item[1], 1) for item in stl_group]
		else:
			curr_stls += stl_group
			if stage == 2:
				process_assembly(curr_stls, stage, unused_items=scatter_selection)
	
	print("Rendering complete!")

if __name__ == "__main__":
	repo_dir = os.getcwd().split('dslab25')[0] + 'dslab25/'
	stl_dir = os.path.join(repo_dir, "assets/vacuum_pump/stl/pieces")
	output_dir = os.path.join(repo_dir, "assets/vacuum_pump/rendered")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	main()
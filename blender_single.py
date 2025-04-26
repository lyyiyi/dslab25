import bpy
import math
import mathutils
import os
from itertools import product

# ───────────────────────────────────────────────
# helpers (unchanged)
# ───────────────────────────────────────────────
def clear_scene():
	bpy.ops.object.select_all(action='SELECT')
	bpy.ops.object.delete(use_global=False)
	for m in bpy.data.meshes:    bpy.data.meshes.remove(m)
	for m in bpy.data.materials: bpy.data.materials.remove(m)

def setup_ground():
	bpy.ops.mesh.primitive_plane_add(size=25, location=(0, 0, 0))
	plane = bpy.context.active_object
	mat = bpy.data.materials.new("GroundMaterial")
	mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
	plane.data.materials.append(mat)
	bpy.ops.rigidbody.object_add()
	plane.rigid_body.type = 'PASSIVE'
	plane.rigid_body.collision_shape = 'CONVEX_HULL'
	plane.rigid_body.use_margin = False
	return plane

def setup_lights():
	bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
	main = bpy.context.active_object
	main.data.energy = 1000
	main.data.size = 5
	radius, z, n = 5, 5, 8
	for i in range(n):
		ang = 2*math.pi*i/n
		x, y = radius*math.cos(ang), radius*math.sin(ang)
		bpy.ops.object.light_add(type='POINT', location=(x, y, z))
		bpy.context.active_object.data.energy = 400

def make_material(p):
	mat = bpy.data.materials.new("Material")
	mat.use_nodes = True
	bsdf = mat.node_tree.nodes.get("Principled BSDF")
	bsdf.inputs["Base Color"].default_value = tuple(p["color"])
	bsdf.inputs["Roughness"].default_value = p["roughness"]
	bsdf.inputs["Metallic"].default_value  = p["metallic"]
	return mat

def import_stl(fp):
	if not os.path.exists(fp):
		print("⚠ File not found:", fp)
		return None
	bpy.ops.wm.stl_import(filepath=fp)
	return bpy.context.selected_objects[0]

def render_image(cam, out_dir, name):
	scene = bpy.context.scene
	scene.camera = cam
	scene.render.filepath = os.path.join(out_dir, name)
	bpy.ops.render.render(write_still=True)

# ───────────────────────────────────────────────
#  CENTRING & SCALING  +  CIRCULAR CAMERAS
# ───────────────────────────────────────────────
def process_assembly(items, stage, perm_id=None):
	scene = bpy.context.scene
	scene.render.engine = 'CYCLES'
	try:
		prefs = bpy.context.preferences.addons['cycles'].preferences
		prefs.compute_device_type = 'METAL'
		scene.cycles.device = 'GPU'
	except Exception:
		pass
	scene.cycles.samples = 32
	scene.cycles.use_adaptive_sampling = True
	scene.cycles.max_bounces = scene.cycles.diffuse_bounces = scene.cycles.glossy_bounces = 2

	clear_scene();	setup_ground();	setup_lights()

	bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0,0,0))
	parent_empty = bpy.context.active_object
	parent_empty.name = "Assembly_Parent"

	objects = []
	for it in items:
		if len(it)==3 and not it[2]:	# skip screws with flag 0
			continue
		obj = import_stl(it[0])
		if obj is None:  continue
		obj.data.materials.clear()
		obj.data.materials.append(make_material(it[1]))
		bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
		obj.parent = parent_empty
		objects.append(obj)

	if not objects:	return

	# ---- world‑space bounding box ---------------------------------
	min_x=min_y=min_z=float('inf');	max_x=max_y=max_z=-float('inf')
	for o in objects:
		for corner in o.bound_box:
			v = o.matrix_world @ mathutils.Vector(corner)
			min_x, max_x = min(min_x,v.x), max(max_x,v.x)
			min_y, max_y = min(min_y,v.y), max(max_y,v.y)
			min_z, max_z = min(min_z,v.z), max(max_z,v.z)

	cx, cy = (min_x+max_x)/2, (min_y+max_y)/2
	parent_empty.location = (-cx, -cy, -min_z)	# XY centre → origin, rest on Z=0

	# ---- uniform scale to fit target size --------------------------
	target = 5.0						# final max dimension in metres
	max_dim = max(max_x-min_x, max_y-min_y, max_z-min_z)
	if max_dim > 0:
		sf = target / max_dim
		parent_empty.scale = (sf, sf, sf)
		parent_empty.location[0] *= sf
		parent_empty.location[1] *= sf
		parent_empty.location[2] *= sf

	# ---- circular camera sweep ------------------------------------
	scene.render.resolution_x = scene.render.resolution_y = 512
	scene.render.image_settings.file_format = 'JPEG'

	cam_count  = 4   # number of cameras around the circle
	cam_radius = 10   # radius of circle (metres)
	cam_height = 18    # Z height of cameras (metres)

	for i in range(cam_count):
		angle = 2 * math.pi * i / cam_count
		x = cam_radius * math.cos(angle)
		y = cam_radius * math.sin(angle)
		z = cam_height

		bpy.ops.object.camera_add(location=(x, y, z))
		cam = bpy.context.active_object

		# point camera at the origin
		vec = mathutils.Vector((0, 0, 0)) - cam.location
		cam.rotation_euler = vec.to_track_quat('-Z', 'Y').to_euler()

		cam.data.lens = 50
		render_image(cam, output_dir, f"{stage}_object_render_{i}.jpg")

# ───────────────────────────────────────────────
# main driver (unchanged)
# ───────────────────────────────────────────────
def main():
	base   = {"color":[.2,.2,.2,1], "roughness":.5,  "metallic":.9}
	screws = {"color":[.4,.4,.4,1], "roughness":.65, "metallic":1}
	axel   = {"color":[.04,.035,.03,1],"roughness":.5,"metallic":.7}
	dark   = {"color":[.1,.1,.1,1], "roughness":.65, "metallic":.9}
	hub    = {"color":[.2,.2,.2,1], "roughness":.65, "metallic":1}
	plate  = {"color":[0,0,0,1],    "roughness":1,   "metallic":0}

	stages = [
		[(os.path.join(stl_dir,"1_base.stl"),base),
		 (os.path.join(stl_dir,"1_base_deckel.stl"),base),
		 (os.path.join(stl_dir,"1_base_deckel_screws.stl"),screws)],
		[(os.path.join(stl_dir,"2_axel.stl"),axel)],
		[(os.path.join(stl_dir,"3_middle_part.stl"),base),
		 (os.path.join(stl_dir,"3_middle.stl"),base)],
		[(os.path.join(stl_dir,"4_diamond.stl"),dark)],
		[(os.path.join(stl_dir,"5_new.stl"),hub)],
		[(os.path.join(stl_dir,"6_screws_01.stl"),screws,True),
		 (os.path.join(stl_dir,"6_screws_02.stl"),screws,True),
		 (os.path.join(stl_dir,"6_screws_03.stl"),screws,True)],
		[(os.path.join(stl_dir,"7_plate.stl"),plate)],
		[(os.path.join(stl_dir,"8_screws_01.stl"),screws,True),
		 (os.path.join(stl_dir,"8_screws_02.stl"),screws,True),
		 (os.path.join(stl_dir,"8_screws_03.stl"),screws,True),
		 (os.path.join(stl_dir,"8_screws_04.stl"),screws,True),
		 (os.path.join(stl_dir,"8_screws_05.stl"),screws,True)]
	]

	for stg, grp in enumerate(stages):
		process_assembly(grp, stg)
	print("✓ Rendering complete")

# ───────────────────────────────────────────────
if __name__ == "__main__":
	repo_dir   = os.getcwd().split('dslab25')[0] + 'dslab25/'
	stl_dir    = os.path.join(repo_dir,"assets/vacuum_pump/stl/pieces")
	output_dir = os.path.join(repo_dir,"assets/vacuum_pump/rendered_single")
	os.makedirs(output_dir, exist_ok=True)
	main()
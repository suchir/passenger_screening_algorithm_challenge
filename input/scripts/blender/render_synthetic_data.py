import bpy
import os
import math
import random
import json


scene = bpy.data.scenes['Scene']
camera = bpy.data.objects['Camera']


def get_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def init_renderer():
    render = scene.render
    render.resolution_x = 256
    render.resolution_y = 330
    render.resolution_percentage = 100
    render.engine = 'CYCLES'
    scene.cycles.samples = 1


def move_camera(angle):
    radius = 1
    camera.location = (radius*math.sin(angle), -radius*math.cos(angle), 0)
    camera.rotation_euler = (math.pi/2, 0, angle)
    camera.rotation_euler = (math.pi/2, 0, angle)


def init_scene():
    camera = bpy.data.cameras['Camera']
    camera.type = 'ORTHO'
    camera.ortho_scale = 2
    camera.draw_size = 0

    bpy.data.worlds['World'].horizon_color = (1, 1, 1)

    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()


def create_nodes(texture_path):
    scene.use_nodes = True
    layers = scene.node_tree.nodes['Render Layers']
    composite = scene.node_tree.nodes['Composite']
    mrange = scene.node_tree.nodes.new('CompositorNodeMapRange')
    mrange.inputs[1].default_value = 0.5
    mrange.inputs[2].default_value = 1.5
    scene.node_tree.links.new(mrange.inputs[0], layers.outputs[2])
    scene.node_tree.links.new(composite.inputs[0], mrange.outputs[0])

    colors = bpy.data.materials.new(name='Zones')
    colors.use_nodes = True
    emission = colors.node_tree.nodes.new('ShaderNodeEmission')
    texture = colors.node_tree.nodes.new('ShaderNodeTexImage')
    bpy.data.images.load(texture_path)
    texture.image = bpy.data.images[texture_path.split('/')[-1]]
    colors_output = colors.node_tree.nodes.get('Material Output')
    colors.node_tree.links.new(colors_output.inputs[0], emission.outputs[0])
    colors.node_tree.links.new(emission.inputs[0], texture.outputs[0])


def import_mesh(filepath):
    bpy.ops.import_scene.makehuman_mhx2(filepath=filepath)

    name = filepath.split('/')[-1].split('.')[0].capitalize()

    bpy.data.objects[name].name = 'mesh'
    bpy.data.objects['%s:Body' % name].name = 'body'
    bpy.data.objects['%s:High-poly' % name].name = 'eyes'

    bones = bpy.data.objects['mesh'].pose.bones
    bones['clavicle_l'].rotation_quaternion = (1, 0, 0, 0.2)
    bones['upperarm_l'].rotation_quaternion = (0.9, 0.4, -0.5, 0.3)
    bones['lowerarm_l'].rotation_quaternion = (1, 0.5, 0.1, 0)
    bones['clavicle_r'].rotation_quaternion = (1, 0, 0, -0.2)
    bones['upperarm_r'].rotation_quaternion = (0.9, 0.4, 0.5, -0.3)
    bones['lowerarm_r'].rotation_quaternion = (1, 0.5, -0.1, 0)

    body = bpy.data.objects['body']
    body.data.materials[0] = bpy.data.materials.get('Zones')

    scene.objects.active = body
    bpy.data.objects['eyes'].select = True
    bpy.ops.object.delete()


def delete_mesh():
    bpy.data.objects['body'].select = True
    bpy.data.objects['mesh'].select = True
    bpy.ops.object.delete()


def get_pose():
    def noisy(pos, amt):
        pos = list(pos)
        for i, x in enumerate(pos):
            pos[i] = pos[i] + random.uniform(-amt, amt)
        return tuple(pos)

    bones = bpy.data.objects['mesh'].pose.bones
    ret = {}
    for bone in bones:
        if any(bone.name.startswith(x) for x in ('clavicle', 'upperarm', 'lowerarm')):
            amt = 0.2
        else:
            amt = 0.05
        ret[bone.name] = noisy(bone.rotation_quaternion, amt)
    return ret


def apply_pose(pose):
    bones = bpy.data.objects['mesh'].pose.bones
    for bone in pose:
        bones[bone].rotation_quaternion = pose[bone]


def delete_mesh():
    bpy.data.objects['body'].select = True
    bpy.data.objects['mesh'].select = True
    bpy.ops.object.delete()


def render_body(filename, num_angles):
    render = bpy.data.scenes['Scene'].render

    poses = [get_pose() for _ in range(num_angles)]
    for i in range(num_angles):
        apply_pose(poses[i])
        move_camera(2*math.pi/num_angles*i)

        scene.use_nodes = True
        render.filepath = os.getcwd() + '/%s_%s_depth.png' % (filename, i)
        bpy.ops.render.render(write_still=True)

        scene.use_nodes = False
        render.filepath = os.getcwd() + '/%s_%s_zones.png' % (filename, i)
        bpy.ops.render.render(write_still=True)


cfg = get_config()

init_renderer()
init_scene()
create_nodes(cfg['texture_path'])

for path in cfg['mesh_paths']:
    filename = path.split('/')[-1].split('.')[0]
    import_mesh(path)
    render_body(filename, cfg['num_angles'])
    delete_mesh()

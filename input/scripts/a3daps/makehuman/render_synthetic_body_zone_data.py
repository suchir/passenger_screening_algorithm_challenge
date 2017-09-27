import bpy
import os
import math
import random
import json


def get_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def init_renderer():
    render = bpy.data.scenes['Scene'].render
    render.resolution_x = 256
    render.resolution_y = 330
    render.resolution_percentage = 100
    render.engine = 'CYCLES'
    bpy.data.scenes['Scene'].cycles.samples = 8


def move_camera(angle):
    bpy.data.objects['Camera'].location = (1.8*math.sin(angle), -1.8*math.cos(angle), 0.8)
    bpy.data.objects['Camera'].rotation_euler = (math.pi/2, 0, angle)
    bpy.data.objects['Lamp'].rotation_euler = (math.pi/2, 0, angle)


def init_scene():
    lamp = bpy.data.lamps['Lamp']
    lamp.type = 'SUN'

    bpy.data.worlds['World'].horizon_color = (0, 0, 0)

    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete() 


def create_materials(texture_path):
    skin = bpy.data.materials.new(name='Skin')
    skin.use_nodes = True
    bsdf = skin.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.3, 0.3, 0.3, 1)
    bsdf.inputs[1].default_value = 1.0
    skin_output = skin.node_tree.nodes.get('Material Output')
    skin.node_tree.links.new(skin_output.inputs[0], bsdf.outputs[0])

    colors = bpy.data.materials.new(name='Colors')
    colors.use_nodes = True
    emission = colors.node_tree.nodes.new('ShaderNodeEmission')
    texture = colors.node_tree.nodes.new('ShaderNodeTexImage')
    bpy.data.images.load(texture_path)
    texture.image = bpy.data.images[texture_path.split('/')[-1]]
    colors_output = colors.node_tree.nodes.get('Material Output')
    colors.node_tree.links.new(colors_output.inputs[0], emission.outputs[0])
    colors.node_tree.links.new(emission.inputs[0], texture.outputs[0])


def get_poses():
    def noisy(pos, amt):
        pos = list(pos)
        for i, x in enumerate(pos):
            pos[i] = pos[i] + random.uniform(-amt, amt)
        return tuple(pos)

    bones = bpy.data.objects['mesh'].pose.bones
    ret = {}
    for bone in bones:
        # amt = 0.25 if any(bone.name.startswith(x) for x in ('clavicle', 'upperarm', 'lowerarm')) \
        #       else 0.1
        ret[bone.name] = noisy(bone.rotation_quaternion, 0.1)
    return ret


def apply_pose(pose):
    bones = bpy.data.objects['mesh'].pose.bones
    for bone in pose:
        bones[bone].rotation_quaternion = pose[bone]


def import_mesh(filepath):
    bpy.ops.import_scene.makehuman_mhx2(filepath=filepath)

    name = filepath.split('/')[-1].split('.')[0].capitalize()

    bpy.data.objects[name].name = 'mesh'
    bpy.data.objects['%s:Body' % name].name = 'body'
    bpy.data.objects['%s:High-poly' % name].name = 'eyes'

    bones = bpy.data.objects['mesh'].pose.bones
    bones['clavicle_l'].rotation_quaternion = (1, 0, 0, 0.2)
    bones['upperarm_l'].rotation_quaternion = (0.7, 0.5, -0.3, 0.3)
    bones['lowerarm_l'].rotation_quaternion = (1, 0.3, 0, 0)
    bones['clavicle_r'].rotation_quaternion = (1, 0, 0, -0.2)
    bones['upperarm_r'].rotation_quaternion = (0.7, 0.5, 0.3, -0.3)
    bones['lowerarm_r'].rotation_quaternion = (1, 0.3, 0, 0)

    bpy.context.scene.objects.active = bpy.data.objects['body']
    bpy.data.objects['eyes'].select = True
    bpy.ops.object.delete()


def apply_skin():
    body = bpy.data.objects['body']
    body.data.materials[0] = bpy.data.materials.get('Skin')


def apply_colors():
    body = bpy.data.objects['body']
    body.data.materials[0] = bpy.data.materials.get('Colors')


def render_body(poses, filename, mode, num_angles):
    render = bpy.data.scenes['Scene'].render

    for i in range(num_angles):
        apply_pose(poses[i])
        move_camera(2*math.pi/num_angles*i)
        render.filepath = os.getcwd() + '/%s_%s_%s.png' % (filename, i, mode)
        bpy.ops.render.render(write_still=True)


def delete_mesh():
    bpy.data.objects['body'].select = True
    bpy.data.objects['mesh'].select = True
    bpy.ops.object.delete()


cfg = get_config()

init_renderer()
init_scene()
create_materials(cfg['texture_path'])

for i, path in enumerate(cfg['mesh_paths']):
    filename = path.split('/')[-1].split('.')[0]
    import_mesh(path)
    poses = [get_poses() for _ in range(cfg['num_angles'])]
    apply_skin()
    render_body(poses, filename, 'skin', cfg['num_angles'])
    apply_colors()
    render_body(poses, filename, 'color', cfg['num_angles'])
    delete_mesh()

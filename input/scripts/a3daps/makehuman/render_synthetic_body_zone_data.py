import bpy
import os
import math
import random


def get_filepaths():
    return open('filepaths.txt', 'r').read().split()


def init_renderer():
    render = bpy.data.scenes['Scene'].render
    render.resolution_x = 256
    render.resolution_y = 330
    render.resolution_percentage = 100
    render.engine = 'CYCLES'
    bpy.data.scenes['Scene'].cycles.samples = 8


def init_scene():
    bpy.data.objects['Camera'].location = (0, -2.5, 1)
    bpy.data.objects['Camera'].rotation_euler = (math.pi/2, 0, 0)

    lamp = bpy.data.lamps['Lamp']
    lamp.type = 'SUN'
    lamp.use_nodes = True
    lamp.node_tree.nodes['Emission'].inputs[1].default_value = 0.25
    bpy.data.objects['Lamp'].rotation_euler = (math.pi/2, 0, 0)

    bpy.data.worlds['World'].horizon_color = (0, 0, 0)

    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete() 


def create_materials(texture_path):
    metal = bpy.data.materials.new(name='Metal')
    metal.use_nodes = True
    bsdf = metal.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Metallic'].default_value = 1.0
    metal_output = metal.node_tree.nodes.get('Material Output')
    metal.node_tree.links.new(metal_output.inputs[0], bsdf.outputs[0])

    colors = bpy.data.materials.new(name='Colors')
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
    body = bpy.data.objects['body']

    def noisy(*pos):
        pos = list(pos)
        for i, x in enumerate(pos):
            pos[i] = pos[i] + random.uniform(-0.1, 0.1)
        return tuple(pos)

    bones['clavicle_l'].rotation_quaternion = noisy(1, 0, 0, 0.2)
    bones['upperarm_l'].rotation_quaternion = noisy(0.8, 0, -0.2, 0.4)
    bones['lowerarm_l'].rotation_quaternion = noisy(0.8, 0.5, -0.2, 0.3)
    bones['clavicle_r'].rotation_quaternion = noisy(1, 0, 0, -0.2)
    bones['upperarm_r'].rotation_quaternion = noisy(0.8, 0, 0.2, -0.4)
    bones['lowerarm_r'].rotation_quaternion = noisy(0.8, 0.5, 0.2, -0.3)
    bpy.context.scene.objects.active = body
    bpy.ops.object.modifier_apply(modifier='ARMATURE')
    bpy.data.objects['mesh'].select = True
    bpy.data.objects['eyes'].select = True
    bpy.ops.object.delete()

    body.lock_rotation = (True, True, False)


def apply_metal():
    body = bpy.data.objects['body']
    body.data.materials[0] = bpy.data.materials.get('Metal')


def apply_colors():
    body = bpy.data.objects['body']
    body.data.materials[0] = bpy.data.materials.get('Colors')


def render_body(filename, mode):
    body = bpy.data.objects['body']
    render = bpy.data.scenes['Scene'].render

    n = 64
    for i in range(n):
        body.rotation_euler = (0, 0, 2*math.pi/n*i)
        render.filepath = os.getcwd() + '/%s_%s_%s.png' % (filename, i, mode)
        bpy.ops.render.render(write_still=True)


def delete_mesh():
    bpy.data.objects['body'].select = True
    bpy.ops.object.delete()


init_renderer()
init_scene()

files = get_filepaths()
texture_path, mesh_paths = files[0], files[1:]

create_materials(texture_path)

for i, path in enumerate(mesh_paths):
    filename = path.split('/')[-1].split('.')[0]
    import_mesh(path)
    apply_metal()
    render_body(filename, 'metal')
    apply_colors()
    render_body(filename, 'color')
    delete_mesh()

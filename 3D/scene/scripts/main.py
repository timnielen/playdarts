import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import pandas as pd
from uuid import uuid4
import os

print("numpy version", np.__version__)
print("pandas version", pd.__version__)

numbers = np.array([20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5], dtype=np.int8)
fields = np.array([*numbers, *(2*numbers), *(3*numbers), 25, 50, 0])
print(fields)

instance_coll = bpy.data.collections.get("tmp")
if not instance_coll:
    instance_coll = bpy.data.collections.new("tmp")
    bpy.context.scene.collection.children.link(instance_coll)

def clean_scene():
    for obj in instance_coll.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    
scene = bpy.context.scene
cam = bpy.data.objects['Camera']


def randomize_board(num_board):
    name = f"board{num_board}"
    obj = bpy.data.objects.get(name)
    if not obj:
        raise ValueError(f"Board with name '{name}' not found.")
    angle = np.random.randint(10) * 0.2 * np.pi 
    obj.rotation_euler[2] = angle
    
def random_in_circle(radius):
    angle = np.random.rand() * 2 * np.pi
    distance = radius * np.sqrt(np.random.rand())
    
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    
    return x, y

def random_in_sector(range_angle=None, scale=None):
    if not scale:
        assert range_angle is not None
        angles = np.random.uniform(-range_angle, range_angle, size=2)
    else: 
        angles = np.random.normal(loc=0, scale=scale, size=2)
    tan_alpha, tan_beta = np.tan(angles)
    direction_z = np.sqrt(1/(tan_alpha*tan_alpha + tan_beta*tan_beta + 1))
    direction_x = tan_alpha * direction_z
    direction_y = tan_beta * direction_z
    return Vector((direction_x, direction_y, direction_z))
    
def randomize_camera(radius, range_angle, range_distance, range_lens):
    center = random_in_circle(radius)
    distance = np.random.uniform(range_distance[0], range_distance[1])
    cam.data.lens = (distance - range_distance[0]) / (range_distance[1]-range_distance[0]) * (range_lens[1]-range_lens[0]) + range_lens[0]
    direction = random_in_sector(range_angle=range_angle)
    distance = distance * np.random.uniform(1, 1.5)
    cam.location = distance * direction + Vector((*center, 0))
    cam.rotation_euler = (-direction).to_track_quat('-Z', 'Z').to_euler()
    
    bpy.context.evaluated_depsgraph_get()
    
def randomize_dart(num_dart, target, scale, range_depth = (0.05, 0.15), target_collection=instance_coll, color=None, previous_direction=None):
    name = f"dart{num_dart}"
    collection = bpy.data.collections.get(name)

    if not collection:
        raise ValueError(f"Collection '{name}' not found.")
    
    location = Vector((*np.random.normal(target, scale=scale), 0))
    direction = random_in_sector(scale=np.radians(15))
    if previous_direction is not None:
        direction = previous_direction + direction
        direction.normalize()
    rotation = direction.to_track_quat('Z').to_euler()
    depth = np.random.uniform(*range_depth)
    
    shaft_location = location + (dart_lengths[num_dart-1] - depth) * direction
    
    for obj in collection.objects:
        print(obj.name)
        instance = bpy.data.objects.new(name=f"tmp_{obj.name}", object_data=obj.data)
        instance.location = location - depth * direction
        instance.rotation_euler = rotation
        target_collection.objects.link(instance)
        if color is not None and obj.name in ["Shaft4", "Flight4", "Flight5"]:
            mat = instance.data.materials[0]
            assert mat.use_nodes == True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Base Color"].default_value = tuple(color)
        
    return location, direction, shaft_location

board_radii = [
    {
        "standard": [1.335, 1.6525, 1.0175],
        "bull": [0.1165, 0, 1.9],
    },
    {
        "standard": [1.3605, 1.657, 1.014],
        "bull": [0.124, 0, 1.9],
    },
]
    
def center_field(field, num_board):
    if field < 60:
        angle = np.pi * (0.5 - (field % 20) * 0.1)
        radius = board_radii[num_board-1]["standard"][field // 20]
    else:
        angle = np.random.rand() * 2*np.pi
        radius = board_radii[num_board-1]["bull"][field-60]
    
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    
    return x, y


print(bpy.context.preferences.addons['cycles'].preferences.compute_device_type)# = 'OPTIX'  # or 'CUDA' or 'OPENCL'
print(bpy.context.preferences.addons['cycles'].preferences)
    

num_scenes = 1000
num_board = 1
index = 0
base_directory = "rendered_dual"
directory = f"{base_directory}/imgs_0/"
if not os.path.isdir("rendered"):
    os.mkdir("rendered")
while os.path.isdir(directory):
    index += 1
    directory = f"{base_directory}/imgs_{index}/"
#full_path = os.path.join(os.getcwd(), directory)
#print(os.getcwd())
os.mkdir(directory)

locations = []
for i in range(4):
    angle = np.pi/2 * (.1 + i)
    distance = 1.7
    location = Vector((distance * np.cos(angle), distance * np.sin(angle), 0))
    locations.append(location)
    
print ("Updated directory:" , os.getcwd())
df_data = []
num_dart_models = 6
dart_lengths = [1.40653, 1.45751, 1.40653, 1.45751, 1.45751, 1.45751]
for num_scene in range(num_scenes):
    clean_scene()
    randomize_camera(radius=.3, range_angle=np.radians(45), range_distance=(4,20), range_lens=(np.random.uniform(4,6), np.random.uniform(25,34)))
    randomize_board(num_board)
    
    my_locations = [world_to_camera_view(scene, cam, location )[:2] for location in locations]
    my_shaft_locations = []

    num_dart = np.random.randint(1, num_dart_models+1)
    scale = np.random.uniform(0.05, 0.2)
    
    # Random color generation
    color = np.random.rand(4)
    color[-1] = 1.0  # Set alpha channel to 1 (opaque)
    if np.random.rand() < 0.1:  # 10% chance to use black
        color[:3] = 0
    
    previous_direction = None
    for num_darts in range(4):
        if num_darts > 0:
            dart_loc, previous_direction, shaft_loc = randomize_dart(num_dart, center_field(target, num_board), scale, color=color, previous_direction=previous_direction)
            my_locations.append(world_to_camera_view(scene, cam, dart_loc)[:2])
            my_shaft_locations.append(world_to_camera_view(scene, cam, shaft_loc)[:2])
            
            if np.random.rand() < .5: #in 50% of cases target a different field next
                target = np.random.randint(len(fields))
        else:
            target = np.random.randint(len(fields))
            
        filename = str(uuid4())
        bpy.context.scene.render.filepath = "//" + directory + filename
        bpy.ops.render.render(write_still=True)

        df_data.append({"filename": filename + ".jpg", "locations": np.array(my_locations), "shaft_locations": np.array(my_shaft_locations), "num_darts": num_darts})
        pd.DataFrame(data=df_data).to_pickle(directory + "labels.pkl")


print("done")
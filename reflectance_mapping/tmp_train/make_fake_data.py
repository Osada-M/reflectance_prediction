import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{DIR}/..')

from mymodel.raytracing import RayTracing
from mymodel.sensor import Sensor
from myutils.dataset import PointCloudDataset, MyDataLoader


## =====================================================================
## 


CUBE = f'{MyDataLoader.DOCKER_DIR}/cad_models/rubik\'s cube (sphere)_colored.ply'
LENGTH = 1<<11
# LENGTH = 2000
ID = 0

ROTATE_RADIUS = 60
XY_SPLIT = 1<<8
Z_SPLIT = 1<<6

SUP = 1<<32
INF = -SUP


TRAIN_TXT = f'{MyDataLoader.DOCKER_DIR}/text/train_FAKE.csv'
TEST_TXT = f'{MyDataLoader.DOCKER_DIR}/text/test_FAKE.csv'
VALID_TXT = f'{MyDataLoader.DOCKER_DIR}/text/validation_FAKE.csv'

TEST_DATA_RATIO = 0.2
VALID_DATA_RATIO = 0.2


## =====================================================================
## 


def calc_ref_from_color(color):

    ref_ratio = sum(int(color[2*i:2*(i+1)], 16) / 255 for i in range(3)) / 3.
    
    return ref_ratio



def maek_fake_reflectance_map(pcd, length=LENGTH, use_random=True):
    
    points = np.asarray(pcd.points)
    ratios = np.zeros(points.shape[0])

    ranges_and_colors = [
        (([0.0, 55.0], [55.0, SUP], [2.0, 55.2]), 'BA0C2F'), # red
        (([55.0, SUP], [0.0, 55.0], [2.0, 55.0]), '009A44'), # green
        (([0.0, 55.0], [0.0, 55.0], [55.0, SUP]), 'FFFFFF'), # white
        (([INF,  0.0], [0.0, 55.0], [2.0, 55.0]), '003DA5'), # blue
        (([0.0, 55.0], [0.0, 55.0], [INF,  2.0]), 'FFD700'), # yellow
        (([0.0, 55.0], [-2.0, 0.0], [2.0, 55.2]), 'FE5000'), # orange
    ]


    for (x_range, y_range, z_range), color in ranges_and_colors:

        indices = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
                (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) & \
                (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        
        if not np.any(indices): continue
        
        ref_ratio = calc_ref_from_color(color)
        
        ratios[indices] += ref_ratio

    if use_random:
    
        ratios += 0.01

        random_ratios = np.random.normal(1, 0.2, (length))
        ratios = np.clip(ratios*random_ratios, 0, 1)

    return ratios

    colors = np.clip(colors * np.repeat(random_colors[:, np.newaxis], 3, axis=1), 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.io.write_point_cloud(CUBE.replace('.ply', 'refmap_FAKE.ply'), pcd)

    return pcd
    


## =====================================================================
## 

sensor = Sensor(sensor_preset='classic')
raytracing = RayTracing()


## =====================================================================
## 


path = MyDataLoader.take_object_list(f'{MyDataLoader.DOCKER_DIR}/text/train_objects.csv')[ID]

pcd, scene = MyDataLoader.load_pcd(path, LENGTH)
center = np.mean(np.asarray(pcd.points), axis=0)

ref_map = maek_fake_reflectance_map(pcd, use_random=True)


## save fake pcd
ref_pcd = o3d.geometry.PointCloud()
ref_pcd.points = pcd.points
ref_pcd.colors = o3d.utility.Vector3dVector(np.repeat(ref_map[:, np.newaxis], 3, axis=1))
o3d.io.write_point_cloud(CUBE.replace('.ply', '_refmap_FAKE.ply'), ref_pcd)


# exit()


## =====================================================================
## 


with open(TRAIN_TXT, 'w') as f:
    print(f'id,coordinate:x,y,z,direction:x,y,z,upward:x,y,z,I_C,min_distance,pruned_length', file=f)
with open(TEST_TXT, 'w') as f:
    print(f'id,coordinate:x,y,z,direction:x,y,z,upward:x,y,z,I_C,min_distance,pruned_length', file=f)
with open(VALID_TXT, 'w') as f:
    print(f'id,coordinate:x,y,z,direction:x,y,z,upward:x,y,z,I_C,min_distance,pruned_length', file=f)


z_degs = np.linspace(0, 80, Z_SPLIT)
zs = np.sin(np.deg2rad(z_degs)) * ROTATE_RADIUS
xy_radiuses = np.sqrt(ROTATE_RADIUS**2 - zs**2)


data_length = Z_SPLIT*XY_SPLIT

train_indices = [0]*int(data_length*(1-TEST_DATA_RATIO-VALID_DATA_RATIO))
test_indices = [1]*int(data_length*TEST_DATA_RATIO)
valid_indices = [2]*int(data_length*VALID_DATA_RATIO)

if data_length - len(train_indices + test_indices + valid_indices) > 0:
    train_indices += [0]*(data_length - len(train_indices + test_indices + valid_indices))

indices = np.asarray(train_indices + test_indices + valid_indices, dtype=np.uint8)
indices = np.random.permutation(indices)

# random_indices = np.random.rand(data_length)
# is_test_data = random_indices < TEST_DATA_RATIO
# is_valid_data = (random_indices >= TEST_DATA_RATIO) & (random_indices < (TEST_DATA_RATIO + VALID_DATA_RATIO))

I_C_max = 0


accurate_ratio = [0]*3


print()

with (open(TRAIN_TXT, 'a') as tr, open(TEST_TXT, 'a') as ts, open(VALID_TXT, 'a') as vl):
    
    txt_files = [tr, ts, vl]
    
    for i, z in enumerate(zs):
        
        xy_radius = xy_radiuses[i]
        
        xy_degs = np.linspace(0, 360, XY_SPLIT+1)[:-1]
        xs = np.cos(np.deg2rad(xy_degs)) * xy_radius
        ys = np.sin(np.deg2rad(xy_degs)) * xy_radius
        
        for j, (x, y) in enumerate(zip(xs, ys)):
            
            index = i*XY_SPLIT+j
            file = txt_files[indices[index]]
            # is_valid = is_valid_data[index]
            

            sensor_coord = np.array([x, y, z], dtype=np.float32) + center
            
            sensor_direction = center - sensor_coord
            sensor_direction /= np.linalg.norm(sensor_direction)

            ## 正規分布に従うノイズを加える
            sensor_coord += np.random.normal(0, 0.5, 3)
            sensor_direction += np.random.normal(0, 0.5, 3)
            
            ## xyの単位ベクトル
            xy_vec = np.array([sensor_direction[0], sensor_direction[1], 0])
            xy_vec /= np.linalg.norm(xy_vec)
            
            ## センサの上向きのベクトル
            z_upward = np.dot(sensor_direction, xy_vec)
            z_up_deg = np.arccos(z_upward)
            if np.isnan(z_up_deg): z_up_deg = 0.
            x_upward = xy_vec[0] * np.sin(z_up_deg)
            y_upward = xy_vec[1] * np.sin(z_up_deg)
            sensor_upward = np.array([x_upward, y_upward, z_upward])
            sensor_upward /= np.linalg.norm(sensor_upward)
            
            ## rotate, translate
            sensor.rotate_from_direction(sensor_direction, sensor_upward)
            sensor.translate(sensor_coord)
            
            point_wise_coef, info = raytracing.simulate_point_wise_coefficient(pcd, scene, sensor)
            dist, length = info
            
            I_C = sum(ref_map * point_wise_coef)
            
            if I_C > I_C_max: I_C_max = I_C

            
            line = f'{ID},{",".join(map(str, sensor_coord))},{",".join(map(str, sensor_direction))},{",".join(map(str, sensor_upward))},{I_C},{dist},{length}'
            
            print(line, file=file)
            accurate_ratio[indices[index]] += 1
            
            print(f'\033[1A{index+1}/{data_length}, {accurate_ratio}{" "*30}')


# print(I_C_max); exit()


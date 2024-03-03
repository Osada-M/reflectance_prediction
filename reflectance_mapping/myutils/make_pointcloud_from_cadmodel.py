import open3d as o3d
import numpy as np

import os

DIR = os.path.dirname(os.path.abspath(__file__))
SUP = 1<<32
INF = -SUP


## =====================================================================
## 


NAME = 'rubik\'s cube (sphere)'

file_name = lambda *format: f'{DIR}/../cad_models/{NAME}{format[1]}.{format[0]}'


# mesh = o3d.io.read_triangle_mesh(file_name('stl'))

# pcd = mesh.sample_points_poisson_disk(number_of_points=2048)

# o3d.io.write_point_cloud(file_name('ply'), pcd)


## =====================================================================
## 


# 点群データの読み込み
pcd = o3d.io.read_point_cloud(file_name('ply', ''))

if pcd.is_empty():
    raise ValueError(f'Empty point cloud: {file_name("ply", "")}')


# NumPy配列に変換
points = np.asarray(pcd.points)
colors = np.zeros_like(points)

print(np.max(points, axis=0))
print(np.min(points, axis=0))


## ルービックキューブ

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
    
    color = list(map(lambda x: int(x, 16) / 255., [color[2*i:2*(i+1)] for i in range(3)]))
    
    colors[indices] *= 0.
    colors[indices] += color

colors += 0.01

random_colors = np.random.normal(1, 0.5, (2048))
colors = np.clip(colors * np.repeat(random_colors[:, np.newaxis], 3, axis=1), 0, 1)

pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud(file_name('ply', '_colored'), pcd)

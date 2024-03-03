import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset


## =====================================================================
## 


class PointCloudDataset(Dataset):
    
    
    ## 点群の座標の標準偏差 [cm]
    COODINATE_STD = 16.
    
    
    def __init__(self, object_paths, object_ids, sensor_info, labels, length=2048):
        
        self.object_paths = object_paths
        self.object_ids = object_ids
        self.sensor_coords, self.sensor_directions, self.sensor_upwards = sensor_info
        self.labels = labels
        
        self.pcds = [None]*len(object_paths)
        self.scenes = [None]*len(object_paths)
        self.centers = [None]*len(object_paths)
        
        for i, path in enumerate(object_paths):
            
            self.pcds[i], self.scenes[i] = MyDataLoader.load_pcd(path, length)
            self.centers[i] = np.mean(np.asarray(self.pcds[i].points), axis=0)
            
        self.number_of_points = len(self.pcds[0].points)
            
    
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, i):
        
        points = np.asarray(self.pcds[self.object_ids[i]].points, dtype=np.float32)
        points -= self.centers[self.object_ids[i]]
        points /= PointCloudDataset.COODINATE_STD
        
        colors = np.asarray(self.pcds[self.object_ids[i]].colors, dtype=np.float32)
        colors /= np.max(colors, axis=0)
        colors *= 2.
        colors -= 1.
        
        points = np.concatenate([points, colors], axis=1)
        points = torch.tensor(points, dtype=torch.float32)
        
        sensor_coord = np.array(self.sensor_coords[i], dtype=np.float32)
        sensor_coord = torch.tensor(sensor_coord, dtype=torch.float32)
        
        sensor_direction = np.array(self.sensor_directions[i], dtype=np.float32)
        sensor_direction /= np.linalg.norm(sensor_direction)
        sensor_direction = torch.tensor(sensor_direction, dtype=torch.float32)
        
        sensor_upword = np.array(self.sensor_upwards[i], dtype=np.float32)
        sensor_upword /= np.linalg.norm(sensor_upword)
        sensor_upword = torch.tensor(sensor_upword, dtype=torch.float32)
        
        labels = self.labels[i]
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return points, (sensor_coord, sensor_direction, sensor_upword), labels


    @staticmethod
    def load_pcd_once(path, length=2048):
        
        pcd = MyDataLoader.load_pcd(path, length, use_scene=False)
        
        points = np.asarray(pcd.points, dtype=np.float32)
        points -= np.mean(points, axis=0)
        points /= PointCloudDataset.COODINATE_STD
        
        colors = np.asarray(pcd.colors, dtype=np.float32)
        
        data = np.concatenate([points, colors], axis=1)
        data = np.expand_dims(data, axis=0)
        data = torch.tensor(data, dtype=torch.float32)
        
        return pcd, data
    

## =====================================================================
## 


class MyDataLoader:
    

    DOCKER_DIR = '/workspace/osada_ws/reflectance_mapping'
    
    
    def __init__(self) -> None:
        pass
    
    
    @staticmethod
    def take_object_list(txt):
        
        with open(txt, 'r') as f:
            lines = f.readlines()
            
        obj_list = []
        
        for i, line in enumerate(map(lambda x: x.rstrip('\n'), lines)):
            if not i: continue
            if not line: continue
            
            _id, path = line.split(',')
            
            obj_list.append(path)
        
        return obj_list
    
    
    @staticmethod
    def take_io_data(txt, skip_num=1):
        
        with open(txt, 'r') as f:
            lines = f.readlines()
        
        lines = lines[::skip_num]
            
        object_ids = []
        sensor_coords = []
        sensor_directions = []
        sensor_upwords = []
        labels = []
        
        for i, line in enumerate(map(lambda x: x.rstrip('\n'), lines)):
            if not i: continue
            if not line: continue
            
            obj_id, coord_x, coord_y, coord_z, direction_x, direction_y, direction_z, upward_x, upward_y, upward_z, I_C, min_distance, pruned_length = line.split(',')
            
            object_ids.append(int(obj_id))
            sensor_coords.append([float(coord_x), float(coord_y), float(coord_z)])
            sensor_directions.append([float(direction_x), float(direction_y), float(direction_z)])
            sensor_upwords.append([float(upward_x), float(upward_y), float(upward_z)])
            labels.append([float(I_C)])
        
        return object_ids, (sensor_coords, sensor_directions, sensor_upwords), labels
    
    
    @staticmethod
    def load_pcd(path, length=2048, use_scene=True):
        
        pcd = o3d.io.read_point_cloud(path)
        
        ## down sampling
        acc_length = len(pcd.points)
        if acc_length != length:
            if acc_length > length:
                indices = np.random.choice(acc_length, length, replace=False)
                sampled_points = np.asarray(pcd.points)[indices]
                sampled_point_cloud = o3d.geometry.PointCloud()
                sampled_point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
            else:
                raise ValueError(f'Length of point cloud is less than {length}: {acc_length}')
        
        pcd.estimate_normals()
        
        if not use_scene:
            return pcd
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)[0]

        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        
        return pcd, scene



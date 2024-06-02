import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

import osada


## =====================================================================
## 


class PointCloudDataset(Dataset):
    
    
    ## 点群の座標の標準偏差 [cm]
    COODINATE_STD = 16.
    
    
    def __init__(self, object_paths, object_ids, sensor_info, labels, length=2048,
                 mode:str='default', dataset_room_size:list=[10., 10., 10.], limit=None):
        
        self.object_paths = object_paths
        self.object_ids = object_ids
        self.labels = labels
        
        if limit is not None:
            
            osada.cprint(f'Warning: limit the number of data to {limit}', 'red')
            
            self.object_paths = self.object_paths[:limit]
            self.object_ids = self.object_ids[:limit]
            self.labels = self.labels[:limit]
            
            sensor_info = sensor_info[:limit]
            
        self.sensor_coords, self.sensor_directions, self.sensor_upwards = sensor_info
        
        self.pcds = [None]*len(object_paths)
        self.scenes = [None]*len(object_paths)
        self.centers = [None]*len(object_paths)
        
        for i, path in enumerate(object_paths):
            
            self.pcds[i], self.scenes[i] = MyDataLoader.load_pcd(path, length)
            self.centers[i] = np.mean(np.asarray(self.pcds[i].points), axis=0)
            
        self.number_of_points = len(self.pcds[0].points)
        
        self.mode = mode
        self.dataset_room_size = np.asarray(dataset_room_size).astype(np.float32)
            
    
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, i):
        
        points = np.asarray(self.pcds[self.object_ids[i]].points, dtype=np.float32)
        
        # points -= self.centers[self.object_ids[i]]
        # points /= PointCloudDataset.COODINATE_STD
        # points /= self.centers[self.object_ids[i]]
        
        if self.mode == 'pointnet':
            scale = points / self.dataset_room_size
        
        points /= np.max(points, axis=0)
        
        # points = PointCloudDataset.rotate_points(points, angle=None)
        
        colors = np.asarray(self.pcds[self.object_ids[i]].colors, dtype=np.float32)
        colors /= np.max(colors, axis=0)
        
        if self.mode == 'pointnet':
            points = np.concatenate([points, colors, scale], axis=1)
        
        elif self.mode == 'transformer':
            points = np.concatenate([points, colors], axis=1)
        
        elif self.mode == 'color_only':
            
            # points *= 2.
            # points -= 1.
            
            colors *= 2.
            colors -= 1.
            
            del points
            points = colors
            del colors
            
        else:
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
    def rotate_points_batch_from_tensor(points, dataset_mode='default', angle:float=None):#, device='cuda'):
        
        if dataset_mode == 'color_only':
            return points
        
        points = points.cpu().numpy()
        coords, other_info = points[:, :, :3], points[:, :, 3:]
        del points
        
        ## (a, b, c) -> (a, b, c-μ)
        centers = np.mean(coords, axis=1, keepdims=True)
        coords -= centers
        
        rotated_coords = np.zeros_like(coords, dtype=np.float32)
        
        for b in range(coords.shape[0]):
            
            if angle is None: angle = np.random.rand()*360.
            
            angle_radians = np.radians(angle)

            Rz = np.array([
                [np.cos(angle_radians), -np.sin(angle_radians), 0],
                [np.sin(angle_radians), np.cos(angle_radians), 0],
                [0, 0, 1]
            ]).astype(np.float32)
            
            rotated_coords[b] += coords[b].dot(Rz) + centers[b]
        
        del coords
        
        rotated_points = np.concatenate([rotated_coords, other_info], axis=-1)
        rotated_points = torch.tensor(rotated_points, dtype=torch.float32)#.to(device)
        
        return rotated_points
    

    @staticmethod
    def load_pcd_once(path, length=2048, mode='default', dataset_room_size=[10., 10., 10.,], device='cuda'):
        
        pcd = MyDataLoader.load_pcd(path, length, use_scene=False)
        
        points = np.asarray(pcd.points, dtype=np.float32)
        
        points /= np.min(points, axis=0)
        
        if mode == 'pointnet':
            scale = points / dataset_room_size
        
        points /= np.max(points, axis=0)
        
        # points = PointCloudDataset.rotate_points(points, angle=None)
        
        colors = np.asarray(pcd.colors, dtype=np.float32)
        colors /= np.max(colors, axis=0)
        
        if mode == 'pointnet':
            points = np.concatenate([points, colors, scale], axis=1)
        
        elif mode == 'color_only':
            
            # points *= 2.
            # points -= 1.
            
            colors *= 2.
            colors -= 1.
            
            del points
            points = colors
            del colors
            
        else:
            points = np.concatenate([points, colors], axis=1)
        
        points = torch.tensor(points, dtype=torch.float32).to(device)
        
        # points -= np.mean(points, axis=0)
        # points /= PointCloudDataset.COODINATE_STD
        
        # colors = np.asarray(pcd.colors, dtype=np.float32)
        
        # data = np.concatenate([points, colors], axis=1)
        # data = np.expand_dims(data, axis=0)
        # data = torch.tensor(data, dtype=torch.float32).to(device)
        
        return pcd, points
    

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
                sampled_colors = np.asarray(pcd.colors)[indices]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(sampled_points)
                pcd.colors = o3d.utility.Vector3dVector(sampled_colors)
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



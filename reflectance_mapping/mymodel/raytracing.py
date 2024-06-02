#! /usr/bin/env python3.9

# import sys
# version = sys.version.split(' ')[0]
# if version != '3.9.18':
#     [print('<!> Please use Python 3.9.18 <!>') for _ in range(10)]
#     print('\nmaybe you can use this:\n % python3.9 ~~~.py\n')
#     exit()


## =====================================================================
## 


import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import osada

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)

from sensor import Sensor


## =====================================================================
## 


class RayTracing:

    

    mkdir = lambda path: os.makedirs(path, exist_ok=True)


   
    def __init__(
        self,
        efficient:bool                  = True,
        accurate_raycast:bool           = False,
        accurate_led:bool               = True,
        accurate_pt:bool                = True,
        without_raycast:bool            = False,
        value_clip:bool                 = False,
        exaggeration_attenuation:bool   = False,
        ):
        '''
        @param efficient:               Occulusionを考慮して効率化するか否か
        @param accurate_raycast:        全てのLEDに対してRay-Castingを行うか否か
        @param accurate_led:            全てのLEDに対して計算するか否か
        '''

        self.efficient = efficient
        self.accurate_led = accurate_led
        self.accurate_pt = accurate_pt
        self.accurate_raycast = accurate_raycast
        self.without_raycast = without_raycast
        self.value_clip = value_clip
        self.exaggeration_attenuation = exaggeration_attenuation
        
        self.config = list(map(int, [efficient, accurate_raycast, accurate_led, accurate_pt, without_raycast, value_clip, exaggeration_attenuation]))
        
        ## 点群の数
        self.NUMBER_OF_POINTS = 100_000
        ## 光の減衰率（線形減衰）
        self.ATTENUATION_FACTOR = [1., 1., 0.]

        self.MODES = ['I_C', 'h_led', 'h_pt', 'intersect', 'effected_rate', 'attenuation', 'likehood', None]
        self.ASSUME_RAYCAST_DIST = 50.
        
        if o3d.core.cuda.is_available():
            print(f"@ CUDA is available")
        else:
            print("@ CUDA is not available")
    
    
    def __call__(self, ):
        pass


    ## =====================================================================
    ## 


    def calc_attenuation(self, distance) -> float:
                
        ## alpha = 1 / (1 + c1*d + c2*d^2)
        
        coef = 1.
            
        attenuation = 1.0 / (1.0 +\
            self.ATTENUATION_FACTOR[0] * distance +\
            self.ATTENUATION_FACTOR[1] * (distance ** 2) +\
            self.ATTENUATION_FACTOR[2] * (distance ** 3))
        
        return attenuation * coef
    
    
    @staticmethod
    def clip_value(values, mode):
        
        values[np.isnan(values)] = RayTracing.VALUE_RANGES[mode]['min']
        
        values -= RayTracing.VALUE_RANGES[mode]['min']
        values /= (RayTracing.VALUE_RANGES[mode]['max'] - RayTracing.VALUE_RANGES[mode]['min'])
        values = np.clip(values, 0, 1)
        
        return values
    
    
    @staticmethod
    def optimize_occulusion_threshold(distance):
        
        th = 0.000001
        
        return th


    ## =====================================================================
    ## 

    
    def simulate_point_wise_coefficient(self, pcd, scene, sensor, debug=False) -> tuple:
        '''
        @func:              センサの出力のみをシミュレートする
        @param pcd:         点群
        @param scene:       Ray-Casting用のシーン
        @param sensor:      Sensorクラスのインスタンス
        @param ref_values:  反射率
        '''
        
        
        ## =====================================================================
        ## 
        
        ## p
        coords = np.array(pcd.points)
        length = len(pcd.points)

        
        ## =====================================================================
        ## ray-cast
        
        
        ## d_min
        dist = np.linalg.norm(coords - sensor.get_ref_point(), axis=1).min()
        occulusion_threshold = self.optimize_occulusion_threshold(dist)
        
        if self.without_raycast:
            occlusions = np.zeros(length, dtype=np.uint8)
        
        else:
            if self.accurate_raycast:
                leds = sensor.get_led_xyz()
            else:
                leds = [sensor.get_ref_point()]
            
            occlusions = np.ones(length, dtype=np.uint8)
            
            for led_xyz in leds:
                
                rays = []

                for i, point in enumerate(coords):
                    
                    direction = led_xyz - point
                    direction /= np.linalg.norm(direction)
                    ray = np.concatenate([point, direction], axis=0)
                    rays.append(ray)
                
                rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
                ans = scene.cast_rays(rays)

                t_hit = ans['t_hit'].numpy()
                geometry_ids = ans['geometry_ids'].numpy()
                occ = np.array(
                    (geometry_ids != o3d.t.geometry.RaycastingScene.INVALID_ID) & (t_hit > occulusion_threshold)
                    )

                occlusions *= occ.astype(np.uint8)
        

        pluned = o3d.geometry.PointCloud()
        pluned.points = o3d.utility.Vector3dVector(coords[occlusions == 0])
        
        pluned_length = len(pluned.points)
        

        ## =====================================================================
        ## 
        
        
        if self.accurate_led:
            sensor.L_0 = sensor.led_length * sensor.L_0_ELEM
        if self.accurate_pt:
            sensor.QUANTUM_EFFICIENCY = sensor.pt_length

        
        P2I_coef = sensor.get_coefficient_P2I() * sensor.AMPLIFICATION_BETA
        point_wise_coefficient = np.zeros(length)

        
        ## 各点
        for i, point in enumerate(coords):
            
            if occlusions[i] == 1: continue
            
            ## n
            ## (|n| = 1)
            normal = pcd.normals[i]
            
            ## L_i
            intensity_in = 0
            
            
            ## =====================================================================
            ## LED
            
            if self.accurate_led:
                gen = zip(sensor.get_led_xyz(), sensor.get_led_normals())
            else:
                gen = [[sensor.get_ref_point()], [sensor.get_ref_normal()]]
                
            for s_xyz, s_normal in gen:
                
                ## 点光源からの入射ベクトル
                ## V = p - l
                light_vec = point - s_xyz
                ## d = |v|
                distance = np.linalg.norm(light_vec)
                ## v = V / d
                ## (|v| = 1)
                light_vec /= distance
                
                ## 減衰を計算
                ## alpha = 1 / (1 + c1*d + c2*d^2)
                attenuation = self.calc_attenuation(distance)
                
                ## S = - v dot n
                # likehood = - np.dot(light_vec, normal)
                
                ## 入射光の強度を計算
                ## phi = acos(s dot v)
                phi = np.arccos(np.dot(s_normal, light_vec))
                h_led = sensor.H_LED(phi)
                
                ## L_i = alpha * L_0 * H_{LED}(phi)
                intensity_in += attenuation * sensor.L_0 * h_led
            
            del gen

            
            ## =====================================================================
            ## PhotoTransistor
            
            if self.accurate_pt:
                gen = zip(sensor.get_pt_xyz(), sensor.get_pt_normals())
            else:
                gen = [[sensor.get_ref_point()], [sensor.get_ref_normal()]]
            
            for s_xyz, s_normal in gen:   
                
                ## 観測点に向かう反射光ベクトル
                ## U = o - p
                to_observation_point = s_xyz - point
                distance = np.linalg.norm(to_observation_point)
                ## u = U / d
                ## |u| = 1
                to_observation_point /= distance  # 正規化
                
                ## 減衰を計算
                attenuation = self.calc_attenuation(distance)
                
                ## (u dot n)
                effected_rate = np.dot(to_observation_point, normal)
                
                ## 反射光の強さを計算（ラムバート反射に基づく）
                ## E_i = max(e_i * R_i, 0)
                effected_rate = max(effected_rate, 0)
                
                ## 観測される光量を計算
                ## psi = acos(u dot n)
                psi = np.arccos(- np.dot(to_observation_point, s_normal))
                ## H_{PT}(psi) = cos^C(psi)
                h_pt = sensor.H_PT(psi)
                
                ## E^{obs}_i = alpha * E_i * H_{PT}(psi)
                observed = attenuation * h_pt * effected_rate
                
                point_wise_coefficient[i] += observed

            del gen

            point_wise_coefficient[i] *= P2I_coef * intensity_in
        
        
        return point_wise_coefficient, (dist, pluned_length,)


    ## =====================================================================
    ## for training

    
    def set_dataset(self, dataset):
        
        self.dataset = dataset
        self.dataset_length = len(self.dataset)
        self.dataset_index = 0
        
        osada.cprint(f'@ set new dataset. length: {self.dataset_length}', 'gray')
    
    
    def forward(self, batch_size, sensor, sensor_info, index=None, device=None):
        
        if index is not None:
            self.dataset_index = index
        
        if self.dataset_index >= self.dataset_length:
            osada.cprint(f'<!> index is out of range. index: {self.dataset_index}, length: {self.dataset_length}', 'red')
            raise IndexError
        
        if device is None: device = torch.device('cuda')
        
        point_wise_coefs = np.zeros((batch_size, self.dataset.number_of_points))

        
        for b in range(batch_size):
            
            sensor_coord = sensor_info[0][b]
            sensor_direction = sensor_info[1][b]
            sensor_upword = sensor_info[2][b]
            
            obj_id = self.dataset.object_ids[self.dataset_index]
            
            sensor.rotate_from_direction(sensor_direction, sensor_upword)
            sensor.translate(sensor_coord)
            
            point_wise_coef, info = self.simulate_point_wise_coefficient(self.dataset.pcds[obj_id], self.dataset.scenes[obj_id], sensor)
            # dist, pruned_length = info

            point_wise_coefs[b,:] = point_wise_coef
            
            self.dataset_index += 1
        
        point_wise_coefs = torch.tensor(point_wise_coefs, dtype=torch.float32).to(device)
        
        return point_wise_coefs
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import osada
import open3d as o3d
import time


from mymodel.nn_model import Backbone, HiddenLayerExtractedModel
from mymodel.raytracing import RayTracing
from mymodel.sensor import Sensor
from myutils.dataset import PointCloudDataset, MyDataLoader
from myutils.myutils import MyUtils, TimeCounter



## =====================================================================
## 


class Test:


    def __init__(self, model_name, weight_name, sensor, raytracing, dataset_room_size=[10., 10., 10.,], backbone=None, device='cuda'):
        
        self.model_name = model_name
        self.weight_name = weight_name
        self.sensor = sensor
        self.raytracing = raytracing
        self.dataset_room_size = dataset_room_size
        
        
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = f'{self.dir}/cad_models'
        self.test_obj = f'{self.dir}/text/test_objects.csv'
        self.test_txt = f'{self.dir}/text/test_FAKE.csv'
        
        self.device = torch.device(device)
        
        self.params = MyUtils.decide_params(self.model_name)
        self.length = self.params['length']
        self.loss = self.params['loss']
        self.dataset_mode = self.params['mode']
        
        self.load_dir = f'{self.dir}/results/{self.model_name}/{self.weight_name}/'
        
        self.mkdir = lambda d: os.makedirs(d, exist_ok=True)
        
        
        if backbone is None:
            self.backbone = Backbone(input_shape=(self.length, 6), model_type=model_name,
                                    is_load_weight=True, weight_path=f'{self.load_dir}/final_epoch.pth')
        else:
            self.backbone = backbone
    
    
    ## =====================================================================
    ## 
        
    
    def test(self, limit=None):
        
        osada.cprint(f'\n\n# {"="*60}\n# test', 'blue')


        with open(f'{self.load_dir}/test.txt', 'w'): pass
        with open(f'{self.load_dir}/test_detail.txt', 'w'): pass


        object_ids_ts, sensor_info_ts, labels_ts = MyDataLoader.take_io_data(self.test_txt, 1)

        self.dataset = PointCloudDataset(
            object_paths=MyDataLoader.take_object_list(self.test_obj),
            object_ids=object_ids_ts,
            sensor_info=sensor_info_ts,
            labels=labels_ts,
            length=self.length,
            mode=self.dataset_mode,
            dataset_room_size=self.dataset_room_size,
            limit=limit,
            )

        self.raytracing.set_dataset(self.dataset)

        dataset_length = len(self.dataset)

        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        self.backbone.model.load_state_dict(torch.load(f'{self.load_dir}/best_validation.pth'))
        self.backbone.model.eval()

        self.metrics = {
            'MSE': 0,
            'MAE': 0,
        }
        
        
        timecounter = TimeCounter(dataset_length)

            
        for i, (points, sensor_info, labels) in enumerate(dataloader):

            with torch.no_grad():
                
                points = PointCloudDataset.rotate_points_batch_from_tensor(points,).to(self.device)
                # points = points.clone().detach().to(dtype=torch.float32).to(self.device)
                labels = labels.clone().detach().to(dtype=torch.float32).to(self.device)
                
                batch_size = len(points)
                
                ## レイトレーシング
                point_wise_coefs = self.raytracing.forward(batch_size=batch_size,
                                                    sensor=self.sensor,
                                                    sensor_info=sensor_info,
                                                    device=self.device,)
                
                ## 順伝播
                I_C_pred = self.backbone.model(points, point_wise_coefs)
                # I_C_pred_numpy = I_C_pred.detach().numpy()
                
                suqare_error = (I_C_pred - labels)**2
                self.metrics['MSE'] += float(suqare_error)
                
                absolute_error = torch.abs(I_C_pred - labels)
                self.metrics['MAE'] += float(absolute_error)
                
                
                with open(f'{self.load_dir}/test_detail.txt', 'a') as f:
                    print(f'{i}, {I_C_pred[0][0]}, {labels[0][0]}, {suqare_error[0][0]}, {absolute_error[0][0]}', file=f)
                
                
                osada.cprint(f'\033[1A  | {i+1} / {dataset_length}  ( {int((i+1)/dataset_length*100)} %  {timecounter.predict_time(i+1)} ),  MAE: {round(float(absolute_error[0][0]), 4)}', 'blue')


        for key, value in self.metrics.items():
            self.metrics[key] = value / dataset_length


        with open(f'{self.load_dir}/test.txt', 'a') as f:
            print(f'MSE(^2): {self.metrics["MSE"]}', file=f)
            print(f'MAE(abs): {self.metrics["MAE"]}', file=f)
            
            
        time_diff = timecounter.second_to_time(time.time() - timecounter.first)
        osada.cprint(f'\033[1A  | total time: {time_diff},  MAE: {round(self.metrics["MAE"], 4)}{" "*30}', 'blue')
        
        
        return self.metrics


    ## =====================================================================
    ## 
    
    
    def output_refmap(self, point_cloud_path=None, mode='default'):
        
        osada.cprint(f'\n\n# {"="*60}\n# view reflectance map', 'blue')

        if point_cloud_path is None:
            point_cloud_path = f'{self.dir}/cad_models/rubik\'s cube (sphere)_colored.ply'

        pcd, data = PointCloudDataset.load_pcd_once(point_cloud_path, length=self.length, 
                                                    mode=self.dataset_mode, dataset_room_size=self.dataset_room_size, 
                                                    device=self.device)

        data = data.unsqueeze(0)
        fake_refmap = torch.ones(1, self.length, 1).to(self.device)


        self.refmap_model = HiddenLayerExtractedModel(self.backbone.model).to(self.device)

        output = self.refmap_model(data, fake_refmap)
        output = output.detach().cpu().numpy()[0]
        
        if mode == 'pointnet':
            ## xyz (a, 1, c) -> (a, c)
            output = output.squeeze(1)
        
        # output[output < 0] = 0.
        # output /= output.max()

        self.refmap_pcd = o3d.geometry.PointCloud()
        self.refmap_pcd.points = o3d.utility.Vector3dVector(pcd.points)
        self.refmap_pcd.colors = o3d.utility.Vector3dVector(MyUtils.array_to_color(output))


        o3d.io.write_point_cloud(f'{self.load_dir}/predicted_refmap.ply', self.refmap_pcd)




## =====================================================================
## 


if __name__ == '__main__':


    # MODEL = 'SimpleLinear'
    # WEIGHT_NAME = '20240317_1926_47'
    
    MODEL = 'PointNet++'
    WEIGHT_NAME = '20240323_1741_47'
    
    # MODEL = 'SimpleTransformer'
    # WEIGHT_NAME = '20240323_1717_16'
    
    sensor = Sensor(sensor_preset='classic')
    raytracing = RayTracing(efficient=True, accurate_raycast=False, accurate_led=True, accurate_pt=True, without_raycast=False)
    
    test = Test(MODEL, WEIGHT_NAME, sensor, raytracing)
    
    # test.test()
    test.output_refmap()
    
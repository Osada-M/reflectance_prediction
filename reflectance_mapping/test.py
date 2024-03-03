import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import osada
import datetime
import open3d as o3d
import time


from mymodel.nn_model import Backbone, HiddenLayerExtractedModel
from mymodel.raytracing import RayTracing
from mymodel.sensor import Sensor
from myutils.dataset import PointCloudDataset, MyDataLoader
from myutils.my_loss_function import ErrorOfCurrent, LossForBackbone, MSE
from myutils.myutils import MyUtils, TimeCounter



DIR = os.path.dirname(os.path.abspath(__file__))


## =====================================================================
## macro


DATASET_DIR = f'{DIR}/cad_models'
TEST_OBJ = f'{DIR}/text/test_objects.csv'
TEST_TXT = f'{DIR}/text/test_FAKE.csv'

LENGTH = 2048

MODEL = 'SimpleLinear'

WEIGHT_NAME = '20240302_1347_57'


## =====================================================================
## other settings


load_dir = f'{DIR}/results/{MODEL}/{WEIGHT_NAME}/'

sensor = Sensor(sensor_preset='classic')
raytracing = RayTracing(efficient=True, accurate_raycast=False, accurate_led=True, accurate_pt=True, without_raycast=False)


mkdir = lambda d: os.makedirs(d, exist_ok=True)


backbone = Backbone(input_shape=(LENGTH, 6), model_type=MODEL,
                    is_load_weight=True, weight_path=f'{DIR}/results/{MODEL}/{WEIGHT_NAME}/final_epoch.pth')


## =====================================================================
## 


osada.cprint(f'\n\n# {"="*60}\n# test', 'blue')


with open(f'{load_dir}/test.txt', 'w'): pass
with open(f'{load_dir}/test_detail.txt', 'w'): pass


object_ids_ts, sensor_info_ts, labels_ts = MyDataLoader.take_io_data(TEST_TXT)

dataset_ts = PointCloudDataset(
    object_paths=MyDataLoader.take_object_list(TEST_OBJ),
    object_ids=object_ids_ts,
    sensor_info=sensor_info_ts,
    labels=labels_ts,
    length=LENGTH,
    )

dataset_length = len(dataset_ts)

dataloader = DataLoader(dataset_ts, batch_size=1, shuffle=True)

backbone.model.load_state_dict(torch.load(f'{load_dir}/best_validation.pth'))
backbone.model.eval()

length_per_epoch = (len(dataset_ts) // 1) + bool(len(dataset_ts) % 1)

metrics = {
    'MSE': 0,
    'MAE': 0,
}

    
for i, (points, sensor_info, labels) in enumerate(dataloader):

    with torch.no_grad():
        
        points = points.clone().detach().to(dtype=torch.float32)
        labels = labels.clone().detach().to(dtype=torch.float32)
        
        batch_size = len(points)
        
        ## レイトレーシング
        point_wise_coefs = raytracing.forward(batch_size=batch_size,
                                              dataset=dataset_ts,
                                              sensor=sensor,
                                              sensor_info=sensor_info,)
        
        ## 順伝播
        I_C_pred = backbone.model(points, point_wise_coefs)
        I_C_pred_numpy = I_C_pred.detach().numpy()
        
        suqare_error = float(((I_C_pred - labels)**2)[0])
        metrics['MSE'] += float(suqare_error)
        
        absolute_error = float(torch.abs(I_C_pred - labels)[0])
        metrics['MAE'] += absolute_error
        
        
        with open(f'{load_dir}/test_detail.txt', 'a') as f:
            print(f'{i}, {I_C_pred_numpy[0][0]}, {float(labels[0])}, {suqare_error}, {absolute_error}', file=f)
        
        
        osada.cprint(f'\033[1A  | {i+1} / {length_per_epoch}  ( {int((i+1)/length_per_epoch*100)} % ),  MAE: {round(absolute_error, 4)}', 'blue')


for key, value in metrics.items():
    metrics[key] = value / length_per_epoch


with open(f'{load_dir}/test.txt', 'a') as f:
    print(f'MSE(^2): {metrics["MSE"]}', file=f)
    print(f'MAE(abs): {metrics["MAE"]}', file=f)


## =====================================================================
## view reflectance map


CUBE = f'{DIR}/cad_models/rubik\'s cube (sphere)_colored.ply'
pcd, data = PointCloudDataset.load_pcd_once(CUBE, 2048)
fake_refmap = torch.ones(1, 2048, 1)


refmap_model = HiddenLayerExtractedModel(backbone.model)

output = refmap_model(data, fake_refmap)
output = output.detach().numpy()[0]
output[output < 0] = 0.
output /= output.max()


refmap_pcd = o3d.geometry.PointCloud()
refmap_pcd.points = o3d.utility.Vector3dVector(pcd.points)
refmap_pcd.colors = o3d.utility.Vector3dVector(MyUtils.array_to_color(output))


o3d.io.write_point_cloud(f'{load_dir}/predicted_refmap.ply', refmap_pcd)


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
NOW = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')


print()


## =====================================================================
## macro


DATASET_DIR = f'{DIR}/cad_models'

TRAIN_OBJ = f'{DIR}/text/train_objects.csv'
TEST_OBJ = f'{DIR}/text/test_objects.csv'
VALID_OBJ = f'{DIR}/text/validation_objects.csv'

TRAIN_TXT = f'{DIR}/text/train_FAKE.csv'
TEST_TXT = f'{DIR}/text/test_FAKE.csv'
VALID_TXT = f'{DIR}/text/validation_FAKE.csv'

LENGTH = 2048

EPOCHS = 100
BATCH_SIZE = 16
DATA_SKIP = 5

# MODEL = 'SimpleLinear'
MODEL = 'SimpleTransformer'

LOSS = 'MSE'
LR = 1e-4

IS_LOAD_WEIGHT = False
WEIGHT_NAME = '20240302_1347_57'


## =====================================================================
## other settings


sensor = Sensor(sensor_preset='classic')
raytracing = RayTracing(efficient=True, accurate_raycast=False, accurate_led=True, accurate_pt=True, without_raycast=False)


mkdir = lambda d: os.makedirs(d, exist_ok=True)


backbone = Backbone(input_shape=(LENGTH, 6), model_type=MODEL, lr=LR,
                    is_load_weight=IS_LOAD_WEIGHT, weight_path=f'{DIR}/results/{MODEL}/{WEIGHT_NAME}/final_epoch.pth')


## =====================================================================
## dataset


object_ids_tr, sensor_info_tr, labels_tr = MyDataLoader.take_io_data(TRAIN_TXT, DATA_SKIP)

dataset_tr = PointCloudDataset(
    object_paths=MyDataLoader.take_object_list(TRAIN_OBJ),
    object_ids=object_ids_tr,
    sensor_info=sensor_info_tr,
    labels=labels_tr,
    length=LENGTH,
    )

dataset_length = len(dataset_tr)
osada.cprint(f'\n# data length: {dataset_length}', 'orange')

if dataset_length%BATCH_SIZE:
    osada.cprint(f'Warning: Hey, your batch size setting is crazy! Lol; {dataset_length} % {BATCH_SIZE} = {dataset_length%BATCH_SIZE}', 'red')
else:
    osada.cprint(f'  | Your batch size setting is perfect!', 'green')

dataloader_tr = DataLoader(dataset_tr, batch_size=BATCH_SIZE, shuffle=True)


object_ids_vl, sensor_info_vl, labels_vl = MyDataLoader.take_io_data(VALID_TXT, DATA_SKIP)
        
dataset_vl = PointCloudDataset(
    object_paths=MyDataLoader.take_object_list(VALID_OBJ),
    object_ids=object_ids_vl,
    sensor_info=sensor_info_vl,
    labels=labels_vl,
    length=LENGTH,
    )

dataloader_vl = DataLoader(dataset_vl, batch_size=BATCH_SIZE, shuffle=True)


## =====================================================================
## loss


# loss_for_backbone = LossForBackbone()
if LOSS == 'MSE':
    loss_for_backbone = MSE()

else:
    raise ValueError(f'Unknown loss function: {LOSS}')


## =====================================================================
## config


# length_per_epoch = (dataset_length // BATCH_SIZE) + bool(dataset_length % BATCH_SIZE)
save_dir = f'{DIR}/results/{MODEL}/{NOW}'
load_dir = f'{DIR}/results/{MODEL}/{WEIGHT_NAME}'

mkdir(f'{DIR}/results')
mkdir(f'{DIR}/results/{MODEL}')
mkdir(f'{save_dir}')
# mkdir(f'{save_dir}/log')
mkdir(f'{save_dir}/weights')

config = {
    'model': MODEL,
    'before_model': WEIGHT_NAME if IS_LOAD_WEIGHT else 'none',
    'number_of_points': LENGTH,
    'dataset_length': dataset_length,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'loss': LOSS,
    'train_txt': TRAIN_TXT,
    'test_txt': TEST_TXT,
    'valid_txt': VALID_TXT,
    'train_obj': TRAIN_OBJ,
    'test_obj': TEST_OBJ,
    'valid_obj': VALID_OBJ,
    'raytracing': ','.join(map(str, raytracing.config)),
    'sensor': sensor.sensor_preset,
    'L_0_elem': sensor.L_0_ELEM,
    'L_0': sensor.L_0,
    'beta': sensor.AMPLIFICATION_BETA,
}

with open(f'{save_dir}/config.txt', 'w') as f:
    for key, value in config.items():
        print(f'{key}: {value}', file=f)


train_loss_progress = []
valid_loss_progress = []


if IS_LOAD_WEIGHT:
    
    with open(f'{load_dir}/loss_train.txt', 'r') as f:

        lines = f.readlines()
        for line in map(lambda x: x.rstrip('\n'), lines):
            epoch, loss = line.split(', ')
            loss = loss.split(': ')[1]
            train_loss_progress.append(float(loss))

    with open(f'{load_dir}/loss_validation.txt', 'r') as f:

        lines = f.readlines()
        for line in map(lambda x: x.rstrip('\n'), lines):
            epoch, loss = line.split(', ')
            loss = loss.split(': ')[1]
            valid_loss_progress.append(float(loss))


## =====================================================================
## train loop


with open(f'{save_dir}/loss_train.txt', 'w'): pass
with open(f'{save_dir}/loss_validation.txt', 'w'): pass


loss_minimum = 1<<32


print(f'\n\n')
for epoch in range(EPOCHS):
    
    osada.cprint(f'\033[3A\n# {"="*60}\n# epoch: {epoch + 1} / {EPOCHS}', 'orange')
    

    ## =====================================================================
    ## train
    
    
    length_per_epoch = (len(dataset_tr) // BATCH_SIZE) + bool(len(dataset_tr) % BATCH_SIZE)
    timecounter = TimeCounter(length_per_epoch)
    
    backbone.model.train()
    loss_sum = 0
    
    print()
    for i, (points, sensor_info, labels) in enumerate(dataloader_tr):

        points = points.clone().detach().to(dtype=torch.float32)
        labels = labels.clone().detach().to(dtype=torch.float32)

        batch_size = len(points)

        ## レイトレーシング
        point_wise_coefs = raytracing.forward(batch_size=batch_size,
                                            dataset=dataset_tr,
                                            sensor=sensor,
                                            sensor_info=sensor_info,)
        
        ## 順伝播
        backbone.optimizer.zero_grad()

        I_C_pred = backbone.model(points, point_wise_coefs)
        I_C_pred_numpy = I_C_pred.detach().numpy()

        
        loss = loss_for_backbone(I_C_pred, labels)
        loss.backward()
        backbone.optimizer.step()

        loss_sum += float(loss.item())
        
        osada.cprint(f'\033[1A  | {i+1} / {length_per_epoch}  x{BATCH_SIZE}  ( {int((i+1)/length_per_epoch*100)} %  {timecounter.predict_time(i+1)} ),  loss: {round(loss.item(), 4)}{" "*10}', 'green')


    loss_sum /= length_per_epoch
    train_loss_progress.append(loss_sum)
    
    time_diff = timecounter.second_to_time(time.time() - timecounter.first)
    osada.cprint(f'\033[1A  | total time: {time_diff},  loss: {round(loss_sum, 4)}{" "*30}', 'green')
    
    with open(f'{save_dir}/loss_train.txt', 'a') as f:
        print(f'epoch: {epoch+1}, {LOSS}: {loss_sum}', file=f)
    
    ## save checkpoint
    if not epoch: mkdir(save_dir)
    torch.save(backbone.model.state_dict(), f'{save_dir}/weights/epoch_{epoch+1}.pth')
    torch.save(backbone.model.state_dict(), f'{save_dir}/final_epoch.pth')

    
    ## =====================================================================
    ## validation
    
    osada.cprint(f'# validation', 'yellow')
    
    backbone.model.eval()
    
    loss_sum = 0
    length_per_epoch = (len(dataset_vl) // BATCH_SIZE) + bool(len(dataset_vl) % BATCH_SIZE)
    timecounter = TimeCounter(length_per_epoch)
        
    with torch.no_grad():
        
        print()
        for i, (points, sensor_info, labels) in enumerate(dataloader_vl):
            
            points = points.clone().detach().to(dtype=torch.float32)
            labels = labels.clone().detach().to(dtype=torch.float32)
            
            batch_size = len(points)
            
            ## レイトレーシング
            point_wise_coefs = raytracing.forward(batch_size=batch_size,
                                                  dataset=dataset_vl,
                                                  sensor=sensor,
                                                  sensor_info=sensor_info,)
            
            ## 順伝播
            I_C_pred = backbone.model(points, point_wise_coefs)
            I_C_pred_numpy = I_C_pred.detach().numpy()
            
            loss = loss_for_backbone(I_C_pred, labels)
            loss_sum += float(loss.item())
            
            osada.cprint(f'\033[1A  | {i+1} / {length_per_epoch}  x{BATCH_SIZE}  ( {int((i+1)/length_per_epoch*100)} %  {timecounter.predict_time(i+1)} ),  loss: {round(loss.item(), 4)}', 'yellow')
    
    
    loss_sum /= length_per_epoch
    valid_loss_progress.append(loss_sum)
    
    time_diff = timecounter.second_to_time(time.time() - timecounter.first)
    osada.cprint(f'\033[1A  | total time: {time_diff},  loss: {round(loss_sum, 4)}{" "*30}', 'yellow')
    
    with open(f'{save_dir}/loss_validation.txt', 'a') as f:
        print(f'epoch: {epoch+1}, {LOSS}: {loss_sum}', file=f)
        
    if loss_sum <= loss_minimum:
        loss_minimum = loss_sum
        torch.save(backbone.model.state_dict(), f'{save_dir}/best_validation.pth')
        osada.cprint(f'\n# loss minimum: {loss_minimum}', 'blue')
    
    MyUtils.plot_loss_progress(train_loss_progress, valid_loss_progress, save_dir, loss=LOSS)
    
    print('\n\n')



## =====================================================================
## test


osada.cprint(f'\n\n# {"="*60}\n# test', 'blue')


with open(f'{save_dir}/test.txt', 'w'): pass
with open(f'{save_dir}/test_detail.txt', 'w'): pass


object_ids_ts, sensor_info_ts, labels_ts = MyDataLoader.take_io_data(TEST_TXT, DATA_SKIP)

dataset_ts = PointCloudDataset(
    object_paths=MyDataLoader.take_object_list(TRAIN_OBJ),
    object_ids=object_ids_ts,
    sensor_info=sensor_info_ts,
    labels=labels_ts,
    length=LENGTH,
    )

dataset_length = len(dataset_ts)

dataloader = DataLoader(dataset_ts, batch_size=1, shuffle=True)

backbone.model.load_state_dict(torch.load(f'{save_dir}/best_validation.pth'))
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
        
        suqare_error = (I_C_pred - labels)**2
        metrics['MSE'] += float(suqare_error)
        
        absolute_error = torch.abs(I_C_pred - labels)
        metrics['MAE'] += float(absolute_error)
        
        
        with open(f'{save_dir}/test_detail.txt', 'a') as f:
            print(f'{i}, {I_C_pred_numpy[0]}, {labels[0]}, {suqare_error[0]}, {absolute_error[0]}', file=f)
        
        
        osada.cprint(f'\033[1A  | {i+1} / {length_per_epoch}  ( {int((i+1)/length_per_epoch*100)} % ),  MAE: {round(absolute_error, 4)}', 'blue')


for key, value in metrics.items():
    metrics[key] = value / length_per_epoch


with open(f'{save_dir}/test.txt', 'a') as f:
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


o3d.io.write_point_cloud(f'{save_dir}/predicted_refmap.ply', refmap_pcd)


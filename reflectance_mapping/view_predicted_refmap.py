import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import osada
import datetime


from mymodel.nn_model import Backbone, HiddenLayerExtractedModel
from mymodel.raytracing import RayTracing
from mymodel.sensor import Sensor
from myutils.dataset import PointCloudDataset, MyDataLoader
from myutils.my_loss_function import ErrorOfCurrent, LossForBackbone, MSE
from myutils.myutils import MyUtils


DIR = os.path.dirname(os.path.abspath(__file__))


## =====================================================================
## macros


CUBE = f'{DIR}/cad_models/rubik\'s cube (sphere)_colored.ply'

MODEL = 'SimpleLinear'
# MODEL = 'SimpleTransformer'
ID = '20240301_1055_14'
EPOCH = 50

WEIGHT = f'{DIR}/results/{MODEL}/{ID}/weights/epoch_{EPOCH}.pth'


## =====================================================================
## setting


pcd, data = PointCloudDataset.load_pcd_once(CUBE, 2048)
fake_refmap = torch.ones(1, 2048, 1)


backbone = Backbone(input_shape=(2048, 6), model_type=MODEL)

for i, (name, module) in enumerate(backbone.model.named_modules()):
    if not i: continue
    print(f'{i:02d} | {name}{" "*(20-len(name))} | {module}')


exit()


## =====================================================================
## extract hidden layer output


backbone.model.load_state_dict(torch.load(WEIGHT))
backbone.model.eval()

refmap_model = HiddenLayerExtractedModel(backbone.model, 'fc_final')

output = refmap_model(data, fake_refmap)
output = output.detach().numpy()[0]
output[output < 0] = 0.
output /= output.max()


## =====================================================================
## save refmap


refmap_pcd = o3d.geometry.PointCloud()
refmap_pcd.points = o3d.utility.Vector3dVector(pcd.points)
refmap_pcd.colors = o3d.utility.Vector3dVector(MyUtils.array_to_color(output))


o3d.io.write_point_cloud(f'{DIR}/results/{MODEL}/{ID}/predicted_refmap.ply', refmap_pcd)


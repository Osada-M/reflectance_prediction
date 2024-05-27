# import sys
# sys.path.append('owl_vit/big_vision/')

import os
# from os.path import expanduser

# import jax
import numpy as np
# import skimage
from skimage import io as skimage_io
import glob
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from tensorflow.keras.models import load_model
import time

import torch
from PIL import Image

import osada
from CLIP import CLIP
from estimation_model import Estimation_Model as est_model
import analyze_result


## ==============================================================================================
osada.cprint('\n@ setting up', 'green')



DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = f'{DIR}/specifications'
SAVE_DIR = f'{DIR}/specifications'

SAVE_NAME = f'{SAVE_DIR}/specifications'


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")
mkdir = lambda x: os.makedirs(x, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    ['clip',    'vit',],
    ['clip',    'resnet',],
    ['resnet',  '',],
    ['vit',     '',],
    ['vgg',     '',],
]

IMAGE_SHAPE = [256, 256]
LOOP = 100



## ==============================================================================================
osada.cprint('\n@ GPU check', 'green')



print(device_lib.list_local_devices())
print()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
   
        
## =====================================================================
## 


mkdir = lambda x: os.makedirs(x, exist_ok=True)

def fill_space(string, length=20):
    return f'{" "*(length-len(string))}{string}'


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params


## ==============================================================================================
osada.cprint('\n@ prediction', 'green')


# general_csv = f'{SAVE_DIR}/general.csv'
mkdir(SAVE_DIR)


data = {}

csv_keys = ['model', 'proccessing time', 'total params', 'trainable params', 'frozen params']

with open(f'{SAVE_NAME}.csv', 'w') as f:
    print(', '.join(csv_keys), file=f)
    
with open(f'{SAVE_NAME}.txt', 'w') as f:
    print(''.join(map(fill_space, csv_keys)), file=f)


for model in MODELS:
    
    backbone_key, clip_backbone = model
    
    ## clip
    if backbone_key == 'clip':
        
        key = f'{backbone_key}_{clip_backbone}'
        osada.cprint(f'\n- {key}', 'yellow')
        
        backbone = None
        
        clip_instance = CLIP(32, clip_backbone)
        
        buf = np.zeros((256, 256, 3))
        buf = Image.fromarray(buf.astype(np.uint8))

        if clip_backbone == 'vit':
            inputs = clip_instance.processor(text=clip_instance.prompts, images=buf, return_tensors="pt", padding=True)
            img = inputs['pixel_values']
            image_size = list(img.size())[1:]
            
        elif clip_backbone == 'resnet':
            inputs = clip_instance.visual.preprocess_images([buf])
            img = np.asarray(inputs[0])
            image_size = list(img.shape)
            

        image = np.zeros((*image_size[1:], image_size[0]))
        
        image = Image.fromarray(image.astype(np.uint8))
        try:
            pred = clip_instance(image, False)
        except TypeError:
            pred = clip_instance([image], False)

        if clip_backbone == 'vit':
            pred = pred.cpu().detach().numpy()
        elif clip_backbone == 'resnet':
            pred = clip_instance.visual.encode(inputs)

        
        output_shape = pred.shape
        

        augment_fn = tf.keras.Sequential([
                layers.Resizing(*image_size[1:]),
            ])
    
    
    ## resnet, vit, etc...
    else:
        
        key = backbone_key
        osada.cprint(f'\n- {key}', 'yellow')
        
        backbone = backbone_key
        
        image_size = [224, 224]
        output_shape = [224, 224, 3]
        
        augment_fn = tf.keras.Sequential([
                layers.Resizing(*image_size),
            ])
        
        if backbone_key == 'vit':
        
            augment_fn = tf.keras.Sequential([
                    layers.Resizing(*image_size),
                    layers.Rescaling(1./255.),
                ])
            output_shape = [3, 224, 224]
        
        # if backbone_key == 'vit':
    
    ## build FC layers
    model_builder = est_model()
    model = model_builder.build_model(mode='mid', output_shape=output_shape, 
                                      batch_size=1, backbone=backbone, backbone_trainable=False,
                                      verbose=False)
    
    data[key] = {
        'time' : [],
        'params_total' : None,
        'params_trainable' : None,
        'params_frozen' : None,
    }
    
    # data[key]['params_total'], data[key]['params_trainable'], data[key]['params_frozen'] = count_parameters(model)
    data[key]['params_total'] = model.count_params()
    osada.cprint(f'  -> trainable : {data[key]["params_trainable"]}, frozen : {data[key]["params_frozen"]}', 'pink')
    
    image = np.ones((*IMAGE_SHAPE, 3))
    
    
    ## =====================================================================
    ## prediction and measurements
    
    
    print()
    for k in range(LOOP):
        
        if backbone_key == 'clip':
        
            img = np.array(augment_fn(image))
            if backbone == 'vit':
                img = img.transpose(2, 0, 1)
                
            img = Image.fromarray(img.astype(np.uint8))
            if clip_instance.backbone == 'vit':
                img = [clip_instance(img, use_text=False).cpu().detach().numpy()]
            elif clip_instance.backbone == 'resnet':
                img = clip_instance.visual.preprocess_images([img])
                img = [clip_instance.visual.encode(img)]
        
        else:
            
            img = np.array(augment_fn(image))
            if backbone_key == 'vit':
                img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
        img = np.array(img)                
        
        ## =================================================================
        start = time.time()
        pred = model.predict(img, verbose=0)[0][0]
        diff = time.time() - start
        ## =================================================================
                    
        data[key]['time'].append(diff)
        
        osada.cprint(f'\033[1A  -> {k+1} / {LOOP} : {diff:.04f} [s]{" "*20}', 'yellow')
    
    osada.cprint(f'\033[1A  -> {k+1} / {LOOP} : completed!{" "*20}', 'yellow')
    
    
    ## =====================================================================
    
    
    data[key]['time'] = np.array(data[key]['time'])
    data[key]['time_mean'] = np.mean(data[key]['time'])
    data[key]['time_median'] = np.median(data[key]['time'])
    
    
    
    # csv_keys = ['model', 'backbone', 'proccessing time', 'trainable params', 'frozen params']
    csv_vals = [
        key,
        data[key]['time_median'],
        data[key]['params_total'],
        data[key]['params_trainable'],
        data[key]['params_frozen'],
    ]
    
    with open(f'{SAVE_NAME}.csv', 'a') as f:
        print(', '.join(map(str, csv_vals)), file=f)
    
    with open(f'{SAVE_NAME}.txt', 'a') as f:
        print(''.join(map(fill_space, map(str, csv_vals))), file=f)


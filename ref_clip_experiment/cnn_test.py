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

import torch
from PIL import Image

import osada
# from CLIP import CLIP
from estimation_model import Estimation_Model as est_model
import analyze_result


# MODEL = 'vgg'


## ==============================================================================================
osada.cprint('\n@ setting up', 'green')


DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


## ==============================================================================================
osada.cprint('\n@ CLIP', 'green')


# if MODEL == 'resnet':
#     backbone = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', pooling=None, classes=1000)
# elif MODEL == 'vgg':
#     backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
# else:
#     raise ValueError('MODEL must be \'resnet\' or \'vgg\'')

# backbone.trainable = False
# backbone.layers[0].trainable = True


## ==============================================================================================
osada.cprint('\n@ take I/O size', 'green')


IMG_SIZE = [224, 224]
# image = np.zeros((1, *IMG_SIZE, 3))
# pred = backbone.predict(image, verbose=False)

OUTPUT_SHAPE = [224, 224, 3]

# OUTPUT_SHAPE = pred.shape[1:]
print(OUTPUT_SHAPE)


## ==============================================================================================
osada.cprint('\n@ dataset', 'green')


train_fold = glob.glob(f'{DIR}/data/train/*')
test_fold = glob.glob(f'{DIR}/data/test/*')

ds_train = []
ds_test = []
gt = [0.327, 0.411, 0.499, 0.546, 0.616, 0.662, 0.754, 0.830, 0.944, 0.965, 0.272, 0.490, 0.612, 0.669]
keys = ['Sauce brown', 'Monster pink', 'Chipstar orange', 'Jelly purple', 'Tea yellow', 'Pringles green', 'Yogurt pink', 'Paperbox blue', 'Pocky yellow', 'Cookie red', 'Chilioil', 'Energy drink', 'Ritz', 'Seaweed']

data = dict(zip(map(lambda x: x.lower().replace(' ', '_'), keys), gt))

for fold in train_fold:
    
    ans = float(fold.replace(f'{DIR}/data/train/', ''))
    images = glob.glob(f'{fold}/*.png')
    ds_train += list(zip(images, [ans] * len(images)))

for image in test_fold:
    
    key = '_'.join(image.replace(f'{DIR}/data/test/', '').split('_')[:-2])
    if not key in data.keys():
        osada.cprint(f'@ \'{key}\' is not in data', 'red')
        continue
    ans = data[key]
    ds_test.append([image, ans, key])



def datagenerator(ds, augment_fn, is_train=False, for_vit=False):
    
    length = len(ds)
    print(length)
    
    if for_vit: osada.cprint(f'@ transform images for ViT', 'green')
    
    if is_train:
       
        for i, (img, ans) in enumerate(ds):
            
            image = np.array(Image.open(img)).astype(np.float32)
            image = np.array(augment_fn(image))
            if for_vit: image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, axis=0)
            # image = backbone.predict(image, verbose=False)
            
            ans = [float(ans)]
            
            yield np.array(image), np.array(ans)
            
    else:
         
        for i, (img, ans, key) in enumerate(ds):
            
            image = np.array(Image.open(img)).astype(np.float32)
            image = np.array(augment_fn(image))
            if for_vit: image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, axis=0)
            # image = backbone.predict(image, verbose=False)
            
            ans = [float(ans)]
            
            yield np.array(image), np.array(ans), key
            

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


## ==============================================================================================
osada.cprint('\n@ test', 'green')


configs = [
    'cnntrain_vit_fine-tuning_20240227_0724_38',
]


for NAME in configs:


    ## ==============================================================================================


    with open(f'{DIR}/results/{NAME}/config.txt', 'r') as f:
        
        lines = f.readlines()
        data = [elem for elem in map(
            lambda x: list(map(lambda y: y.replace(' ', '').replace('\n', ''), x.split(':'))), lines
            )]
        data = dict(data)

    print(data)

    MODE = data['mode']
    META_CONFIG = dict(list(map(lambda x: x.split('='), data['meta_config'].split(','))))
    META_CONFIG = dict(list(map(lambda x: (x[0], bool(x[1] == 'True')), META_CONFIG.items())))
    BACKBONE = data['model']
    
    if BACKBONE == 'vit':
        OUTPUT_SHAPE = [3, 224, 224]

    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(*IMG_SIZE),
        layers.Rescaling(1./255),
    ])

    try:
        model_builder = est_model()
        model = model_builder.build_model(mode=MODE, output_shape=OUTPUT_SHAPE, meta_config=META_CONFIG,
                                        backbone=BACKBONE, backbone_trainable=False)
        model.load_weights(f'{DIR}/results/{NAME}/weights.h5')
    
    except:
        model = load_model(f'{DIR}/results/{NAME}/model.h5')
        # model = load_model(f'{DIR}/results/{NAME}')
    
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.summary()

    # model.load_weights(f'{DIR}/results/{NAME}/weights.h5')

    for is_train, ds in enumerate([ds_test, ds_train]):

        print('>>', ['test', 'train'][is_train])
        
        datagen = datagenerator(ds, resize_and_rescale, is_train=is_train, for_vit=BACKBONE=='vit')
        length = len(ds)

        result = {
            'key' : [],
            'ans' : [],
            'pred': [],
            'diff' : [],
        }

        print()
        
        if is_train:
            
            for i, (img, ans) in enumerate(datagen):
                
                if i >= length: break
                
                pred = model.predict(img, verbose=0)
                
                ans = ans[0]
                pred = pred[0][0]
                
                # result['key'].append(key)
                result['ans'].append(ans)
                result['pred'].append(pred)
                result['diff'].append(pred - ans)
                
                print(f'\033[1A{i+1} / {length} : {ans} {pred}{" "*10}')
                
            with open(f'{DIR}/results/{NAME}/result_detail{["", "_train"][is_train]}.txt', 'w') as f:
                for key, vals in result.items():
                    print(f'{key}{" "*(10 - len(key))}: {" ".join(map(str, vals))}', file=f)

            for key in result.keys():
                result[key] = np.array(result[key])

            mae = np.mean(np.abs(result['diff']))
            std = np.std(result['diff'])

            
        else:
                
            for i, (img, ans, key) in enumerate(datagen):
                
                if i >= length: break
                
                pred = model.predict(img, verbose=0)
                
                ans = ans[0]
                pred = pred[0][0]
                
                result['key'].append(key)
                result['ans'].append(ans)
                result['pred'].append(pred)
                result['diff'].append(pred - ans)
                
                print(f'\033[1A{i+1} / {length} : {ans} {pred}{" "*10}')
                

            with open(f'{DIR}/results/{NAME}/result_detail{["", "_train"][is_train]}.txt', 'w') as f:
                for key, vals in result.items():
                    print(f'{key}{" "*(10 - len(key))}: {" ".join(map(str, vals))}', file=f)

            for key in result.keys():
                result[key] = np.array(result[key])

            mae = np.mean(np.abs(result['diff']))
            std = np.std(result['diff'])

            # print(result)

            analyze_result.call(DIR, NAME, is_train)


        with open(f'{DIR}/results/{NAME}/result.txt{["", "_train"][is_train]}', 'w') as f:
            print(f'''\
mae : {mae}
std : {std}\
''', file=f)

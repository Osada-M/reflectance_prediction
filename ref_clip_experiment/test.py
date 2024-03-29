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

import torch
from PIL import Image

import osada
from CLIP import CLIP
from estimation_model import Estimation_Model as est_model
import analyze_result


## ==============================================================================================
osada.cprint('\n@ setting up', 'green')


DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


## ==============================================================================================
osada.cprint('\n@ CLIP', 'green')


clip_instance = CLIP(32, 'resnet')


## ==============================================================================================
osada.cprint('\n@ take I/O size', 'green')


buf = np.zeros((256, 256, 3))
buf = Image.fromarray(buf.astype(np.uint8))

if clip_instance.backbone == 'vit':
    inputs = clip_instance.processor(text=clip_instance.prompts, images=buf, return_tensors="pt", padding=True)
    img = inputs['pixel_values']
    IMG_SIZE = list(img.size())[1:]
    
elif clip_instance.backbone == 'resnet':
    inputs = clip_instance.visual.preprocess_images([buf])
    img = np.asarray(inputs[0])
    IMG_SIZE = list(img.shape)
    
print(IMG_SIZE)

image = np.zeros((*IMG_SIZE[1:], IMG_SIZE[0]))
image = Image.fromarray(image.astype(np.uint8))
pred = clip_instance([image], False)

if clip_instance.backbone == 'vit':
    pred = pred.cpu().detach().numpy()
elif clip_instance.backbone == 'resnet':
    pred = clip_instance.visual.encode(inputs)

OUTPUT_SHAPE = pred.shape
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
    

# with open(f'{DIR}/data/test.txt', 'w') as f:
#     for img, ans in ds_test:
#         print(f'{img} {ans}', file=f)


def datagenerator(ds, clip_instance, augment_fn, model_aug, is_train=False):
    
    length = len(ds)
    print(length)
    
    if is_train:
       
        for i, (img, ans) in enumerate(ds):
            
            image = skimage_io.imread(img).astype(np.float32)
            image = np.array(augment_fn(image))
            image = Image.fromarray(image.astype(np.uint8))
            if clip_instance.backbone == 'vit':
                image = [clip_instance(image, **model_aug).cpu().detach().numpy()]
            elif clip_instance.backbone == 'resnet':
                image = clip_instance.visual.preprocess_images([image])
                image = [clip_instance.visual.encode(image)]
            # image = [owl_vit(image)]
            
            ans = [float(ans)]
            
            yield np.array(image), np.array(ans)
            
    else:
         
        for i, (img, ans, key) in enumerate(ds):
            
            image = skimage_io.imread(img).astype(np.float32)
            image = np.array(augment_fn(image))
            image = Image.fromarray(image.astype(np.uint8))
            if clip_instance.backbone == 'vit':
                image = [clip_instance(image, **model_aug).cpu().detach().numpy()]
            elif clip_instance.backbone == 'resnet':
                image = clip_instance.visual.preprocess_images([image])
                image = [clip_instance.visual.encode(image)]
            # image = [owl_vit(image)]
            
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


config = [
    # 'clip_resnet_20240213_0739_49',
    # 'clip_resnet_20240213_1207_15',
    # 'clip_resnet_20240213_1634_43',
    'clip_resnet_20240214_0525_19',
    'clip_resnet_20240214_1410_55',
    'clip_resnet_20240214_2255_21',
    'clip_resnet_20240215_0738_09',
    
]
MODEL_AUG = {
    'use_text' : False
}


for NAME in config:


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

    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(*IMG_SIZE[1:]),
        # layers.Rescaling(1./255),
    ])

    model_builder = est_model()
    model = model_builder.build_model(mode=MODE, output_shape=OUTPUT_SHAPE, meta_config=META_CONFIG)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.summary()

    model.load_weights(f'{DIR}/results/{NAME}/weights.h5')

    for is_train, ds in enumerate([ds_test, ds_train]):

        print('>>', ['test', 'train'][is_train])
        
        datagen = datagenerator(ds, clip_instance, resize_and_rescale, model_aug=MODEL_AUG, is_train=is_train)
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

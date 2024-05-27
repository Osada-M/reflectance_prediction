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
from CLIP import CLIP
from estimation_model import Estimation_Model as est_model
import analyze_result


## ==============================================================================================
osada.cprint('\n@ setting up', 'green')



DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = f'{DIR}/folder_wise_test/data'
WEIGHTS_DIR = f'{DIR}/backups'
SAVE_DIR = f'{DIR}/folder_wise_test/results'

ANSWER_CSV = f'{DIR}/answers.csv'


unseens_folders = glob.glob(f'{IMAGES_DIR}/unseen/*')
known_folders = glob.glob(f'{IMAGES_DIR}/known/*')

if len(unseens_folders) == 0 or len(known_folders) == 0:
    raise ValueError('No folder found')


def csv_path(is_seen, model_name):
    
    return f'{SAVE_DIR}/{["unseen", "known"][is_seen]}/{model_name}.csv'


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")
mkdir = lambda x: os.makedirs(x, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    ['clip',    'vit',      '20230827_0945_14'],
    ['clip',    'resnet',   'clip_resnet_20240215_0738_09'],
    # ['resnet',  '',         'resnet_20230628_2217_17'],
    ['resnet',  '',         'cnntrain_resnet_transfer_20240228_1527_05'],
    ['vit',     '',         'vit_20230825_2134_49'],
    ['vgg',     '',         'cnntrain_vgg_transfer_20240226_1010_09'],
]

OMIT_FOLDERS = []

IS_RAW = [
    # 'repellent',
    # 'zone',
    # 'ritz',
    'bond',
    'ritz',
    'zone',
    'water',
]


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


answers = dict()

with open(ANSWER_CSV, 'r') as f:
    lines = f.readlines()

for line in map(str.strip, lines):
    name, ans = line.split(',')
    answers[name] = float(ans)


## ==============================================================================================
osada.cprint('\n@ prediction', 'green')


general_csv = f'{SAVE_DIR}/general.csv'

with open(general_csv, 'w') as f_g:
    print('model, unseen mean, unseen std, known mean, known std', file=f_g)


for model in MODELS:
    
    backbone_key, clip_backbone, weights_id = model
    
    ## clip
    if backbone_key == 'clip':
        osada.cprint(f'\n- {backbone_key} ( {clip_backbone} )', 'yellow')
        
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
        
        osada.cprint(f'\n- {backbone_key}', 'yellow')
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
    
    ## load weights
    try:
        model.load_weights(f'{WEIGHTS_DIR}/{weights_id}/weights.h5')
    
    except:
        model = load_model(f'{WEIGHTS_DIR}/{weights_id}/model.h5')
    
    
    ## =====================================================================
    ## prediction
    
    
    general_data = [[0]*2 for _ in range(2)]
    
    
    for is_seen, folders in enumerate([unseens_folders, known_folders]):
        
        
        name = ['unseen', 'known'][is_seen]
        max_j = len(folders)
        
        csv = csv_path(is_seen, f'{backbone_key}{f"_{clip_backbone}" if clip_backbone else ""}')
        
        errors = []        
        
        with open(csv, 'w') as f:
        
            osada.cprint(f'  -> {name}', 'yellow')
            
            
            for j, folder in enumerate(folders):
                
                object_name = folder.split('/')[-1]
                if object_name in OMIT_FOLDERS: continue
                ans = answers[object_name]
                
                
                imgs = glob.glob(f'{folder}/*.png') + glob.glob(f'{folder}/*.PNG')
                max_k = len(imgs)
                
                predicts_buf = [0]*max_k
                
                for k, img in enumerate(imgs):
                    
                    # object_name = img.split('/')[-2]
                    # if not is_seen and not object_name in IS_RAW:
                    #     luminance_coef = 0.85
                    # else:
                    #     luminance_coef = 1.0
                    
                    if backbone_key == 'clip':
                    
                        img = skimage_io.imread(img).astype(np.float32)
                        img = np.array(augment_fn(img))#  * luminance_coef
                        if backbone == 'vit':
                            img = img.transpose(2, 0, 1)
                            
                        img = Image.fromarray(img.astype(np.uint8))
                        if clip_instance.backbone == 'vit':
                            img = [clip_instance(img, use_text=False).cpu().detach().numpy()]
                        elif clip_instance.backbone == 'resnet':
                            img = clip_instance.visual.preprocess_images([img])
                            img = [clip_instance.visual.encode(img)]
                    
                    else:
                        
                        img = np.array(Image.open(img)).astype(np.float32)
                        img = np.array(augment_fn(img))#  * luminance_coef
                        if backbone_key == 'vit':
                            img = img.transpose(2, 0, 1)
                        img = np.expand_dims(img, axis=0)
                    img = np.array(img)                
                    
                    pred = model.predict(img, verbose=0)[0][0]
                    
                    ## =====================================================================

                    ## =====================================================================
                    
                    errors.append(abs(pred - ans))
                    predicts_buf[k] = pred
                    
                    osada.cprint(f'\033[1A  -> {name} : {j+1} / {max_j} : {k+1} / {max_k}{" "*20}', 'yellow')
                
                print(f'{object_name}, {", ".join(map(str, predicts_buf))}', file=f)
                del predicts_buf
                
                osada.cprint(f'\033[1A  -> {name} : {j+1} / {max_j} : completed!{" "*20}', 'yellow')
    

        errors = np.array(errors)
        general_data[is_seen][0] = np.mean(errors)
        general_data[is_seen][1] = np.std(errors)
    
    with open(general_csv, 'a') as f_g:
        
        print(f'{backbone_key}{f"_{clip_backbone}" if clip_backbone else ""}, {general_data[0][0]}, {general_data[0][1]}, {general_data[1][0]}, {general_data[1][1]}', file=f_g)

exit()


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

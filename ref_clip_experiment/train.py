import os
import numpy as np
import datetime
import glob
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib

import torch
from PIL import Image

import osada
from CLIP import CLIP
from estimation_model import Estimation_Model as est_model
from prompts_for_train import prompts


## ==============================================================================================
osada.cprint('\n@ setting up', 'green')


DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_TEXT = True
BACKBONE = 'vit'

# MULTI_MODAL_MODE = 'add'
# MULTI_MODAL_MODE = 'concat'
MULTI_MODAL_MODE = 'text-only'

## ==============================================================================================
osada.cprint('\n@ CLIP', 'green')


clip_instance = CLIP(32, backbone=BACKBONE, device=DEVICE, is_save_feature=False,
                     use_text=USE_TEXT, multi_modal_mode=MULTI_MODAL_MODE)

# working_test = {
#     'image' : f'{DIR}/tmp/dog.jpg',
#     'prompt' : ['dog', 'cat']
#     }

# probs = clip_instance.contrast(**working_test)
# print(working_test['prompt'])
# print(probs)


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
    
# print(img.shape)

print(IMG_SIZE)

# IMG_SIZE = [3, 224, 224]
image = np.zeros((*IMG_SIZE[1:], IMG_SIZE[0]))
image = Image.fromarray(image.astype(np.uint8))
pred = clip_instance(image, text='this is a test prompt')
# pred = clip_instance([image], text='this is a test prompt', use_text=USE_TEXT)

if clip_instance.backbone == 'vit':
    pred = pred.cpu().detach().numpy()
elif clip_instance.backbone == 'resnet':
    pred = clip_instance.visual.encode(inputs)

OUTPUT_SHAPE = pred.shape
print(OUTPUT_SHAPE)


# inputs = clip_instance.processor(text=[''], images=image, return_tensors="pt", padding=True)
# image = np.array(inputs['pixel_values'])
# OUTPUT_SHAPE = image.shape
# OUTPUT_SHAPE = [224, 224, 3]


## ==============================================================================================
# osada.cprint('\n@ mkdir', 'green')


# now = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
# print(now)

# name = f'{now}'

# try:
#     os.makedirs(f'{DIR}/results/{name}')
#     os.makedirs(f'{DIR}/results/{name}/checkpoints')
# except:
#     pass


## ==============================================================================================
osada.cprint('\n@ dataset', 'green')


train_fold = glob.glob(f'{DIR}/data/train/*')
test_fold = glob.glob(f'{DIR}/data/test/*')
train_prompts = prompts.train_detail
test_prompts = prompts.test_detail

ds_train = []
ds_test = []

gt = [0.327, 0.411, 0.499, 0.546, 0.616, 0.662, 0.754, 0.830, 0.944, 0.965, 0.272, 0.490, 0.612, 0.669]
keys = ['Sauce brown', 'Monster pink', 'Chipstar orange', 'Jelly purple', 'Tea yellow', 'Pringles green', 'Yogurt pink', 'Paperbox blue', 'Pocky yellow', 'Cookie red', 'Chilioil', 'Energy drink', 'Ritz', 'Seaweed']

data = dict(zip(map(lambda x: x.lower().replace(' ', '_'), keys), gt))


for fold in train_fold:
    
    ans = float(fold.replace(f'{DIR}/data/train/', ''))
    images = glob.glob(f'{fold}/*.png')
    prompt = train_prompts[ans]
    
    ds_train += list(zip(images, [prompt]*len(images), [ans]*len(images)))

for image in test_fold:
    
    key = '_'.join(image.replace(f'{DIR}/data/test/', '').split('_')[:-2])
    if not key in data.keys():
        osada.cprint(f'@ \'{key}\' is not in data', 'red')
        continue
    ans = data[key]
    prompt = test_prompts[key]
    
    ds_test.append([image, prompt, ans])



def datagenerator(ds, clip_instance, augment_fn, epochs=1, batch_size=16, shuffle=True, augment=1, break_limit=False, test=False,
                  use_clip=True):

    ds = ds * augment
    
    if shuffle: ds = np.random.permutation(ds)
    
    if test:
        limit = batch_size
    else:
        length = len(ds)
        limit = length - (length % batch_size)
    
    ds = ds[:limit]
    length = len(ds)
    print(length)
    
    if test:
        for _ in range(epochs):
            for i, (img, prm, ans) in enumerate(ds):
                if i >= length:break
                
                image = np.array(Image.open(img))
                image = np.array(augment_fn(image))
                
                if use_clip:
                    image = Image.fromarray(image.astype(np.uint8))
                    image = [clip_instance(image, text=prm,).cpu().detach().numpy()]
                
                ans = [float(ans)]
                
                yield np.array(image), np.array(ans)
        
    else:
        images, answers = [], []
        for _ in range(epochs):
            for i, (img, prm, ans) in enumerate(ds):
                if break_limit and i >= length:break
                if i and not len(answers)%batch_size:
                    yield np.array(images), np.array(answers)
                    images, answers = [], []
                
                image = np.array(Image.open(img))
                image = np.array(augment_fn(image))
                
                if use_clip:
                    image = Image.fromarray(image.astype(np.uint8))
                    if clip_instance.backbone == 'vit':
                        image = clip_instance(image, text=prm,).cpu().detach().numpy()
                    elif clip_instance.backbone == 'resnet':
                        image = clip_instance.visual.preprocess_images([image])
                        image = clip_instance.visual.encode(image)
                    
                
                ans = [float(ans)]
                
                images.append(image)
                answers.append(ans)
        

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
osada.cprint('\n@ set preference', 'green')


configs = [
    
    {'batch_size' : 32, 'epochs' : 200, 'mode' : 'mid', 'load_weights' : False, 'load_id' : ''},
    # {'batch_size' : 32, 'epochs' : 200, 'mode' : 'mid', 'load_weights' : True, 'load_id' : 'clip_resnet_20240213_2100_11'},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'mid', 'load_weights' : True, 'load_id' : '20230827_0809_17'},
    {'batch_size' : 32, 'epochs' : 200, 'mode' : 'mid', 'load_weights' : True, 'load_id' : 'previous'},
    {'batch_size' : 32, 'epochs' : 200, 'mode' : 'mid', 'load_weights' : True, 'load_id' : 'previous'},
    # {'batch_size' : 32, 'epochs' : 200, 'mode' : 'mid', 'load_weights' : True, 'load_id' : 'previous'},
    
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'min', 'load_weights' : False, 'load_id' : ''},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'min', 'load_weights' : True, 'load_id' : 'previous'},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'min', 'load_weights' : True, 'load_id' : 'previous'},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'min', 'load_weights' : True, 'load_id' : 'previous'},
    
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'mid_2', 'load_weights' : False, 'load_id' : ''},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'mid_2', 'load_weights' : True, 'load_id' : 'previous'},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'mid_2', 'load_weights' : True, 'load_id' : 'previous'},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'mid_2', 'load_weights' : True, 'load_id' : 'previous'},
    
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'mid_2', 'load_weights' : False, 'load_id' : ''},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'mid_2', 'load_weights' : True, 'load_id' : 'previous'},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'mid_2', 'load_weights' : True, 'load_id' : 'previous'},
]


previous = None
# buf = 'text_concat_clip_vit_'
buf = 'text_only_clip_vit_'

# BATCH_SIZE = 32
# EPOCHS = 100
# MODE = 'mid'

META_CONFIG = {
    'layernorm' : False,
    'batchnorm' : True,
    'dropout'   : True,
    }
# MODEL_AUG = {
#     'use_text' : False
# }

# LOAD_WEIGHTS = True
# LOAD_ID = '20230626_0322_10'


AUGMENT = 4
LR = 1e-4


if clip_instance.use_text:
    osada.cprint('\n# Use text prompt.', 'lightgreen')
else:
    osada.cprint('\n# DO NOT use text prompt.', 'red')


for elem in configs:
    
    key, val = elem.keys(), elem.values()
    BATCH_SIZE, EPOCHS, MODE, LOAD_WEIGHTS, LOAD_ID = val
    
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
    print(now)

    name = f'{buf}{now}'

    try:
        os.makedirs(f'{DIR}/results/{name}')
        os.makedirs(f'{DIR}/results/{name}/checkpoints')
    except:
        pass
    
    
    if LOAD_WEIGHTS and LOAD_ID == 'previous': LOAD_ID = previous
    previous = name
    
    
    ## ==============================================================================================
    osada.cprint('\n@ train', 'green')


    if LOAD_WEIGHTS:
        with open(f'{DIR}/results/{LOAD_ID}/config.txt', 'r') as f:
            lines = f.readlines()
            ep_buf = int(lines[1].split(': ')[-1])
        LR = 10 ** (- (ep_buf // 100) - 3)
        osada.cprint(f'\n@ load weights : {LOAD_ID}, ep_buf : {ep_buf}', 'orange')
    else:
        ep_buf = 0
        LR = 1e-4

    osada.cprint(f'learning late : {LR}', 'orange')

    with open(f'{DIR}/results/{name}/config.txt', 'w') as f:
        print(f'''\
batch size  : {BATCH_SIZE}
epochs      : {EPOCHS+ep_buf}
mode        : {MODE}
meta_config : {",".join([f"{key}={val}" for key, val in META_CONFIG.items()])}
load_path   : {LOAD_ID if LOAD_WEIGHTS else 'none'}\
''',
file=f)


    train_aug = tf.keras.Sequential([
        layers.Resizing(*IMG_SIZE[1:]),
        # layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        # layers.RandomBrightness(0.2),
    ])
    test_aug = tf.keras.Sequential([
        layers.Resizing(*IMG_SIZE[1:]),
        # layers.Rescaling(1./255),
    ])

    train_datagen = datagenerator(ds_train, clip_instance, train_aug, epochs=EPOCHS,
                                batch_size=BATCH_SIZE, augment=AUGMENT, 
                                use_clip=True)
    # test_datagen = datagenerator(ds_test, clip_instance, test_aug, epochs=EPOCHS, batch_size=16, test=True)


    opt = tf.keras.optimizers.Adam(learning_rate=LR)

    model_builder = est_model()
    model = model_builder.build_model(mode=MODE, output_shape=OUTPUT_SHAPE, meta_config=META_CONFIG,
                                    batch_size=BATCH_SIZE)

    if LOAD_WEIGHTS:
        osada.cprint(f'\n@ load weights : {LOAD_ID}', 'orange')
        model.load_weights(f'{DIR}/results/{LOAD_ID}/weights.h5')
        # model = load_model(f'{DIR}/results/{LOAD_ID}/weights.h5')

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])#, run_eagerly=False)
    model.summary()
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{DIR}/results/{name}/checkpoints',
        save_weights_only=True,
        verbose=0,
        )


    model.fit_generator(train_datagen,
            # validation_data=test_datagen,
            steps_per_epoch=len(ds_train)*AUGMENT//BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[cp_callback],)
    model.save(f'{DIR}/results/{name}/model.h5')
    model.save_weights(f'{DIR}/results/{name}/weights.h5')

    history = model.history.history
    with open(f'{DIR}/results/{name}/history.txt', 'w') as f:
        print(history, file=f)
    
    del model

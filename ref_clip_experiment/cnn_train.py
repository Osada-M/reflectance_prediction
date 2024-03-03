import os
import numpy as np
import datetime
import glob
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.client import device_lib

import torch
from PIL import Image
import time

import osada
from estimation_model import Estimation_Model as est_model


# BACKBONE = 'vgg'


## ==============================================================================================
osada.cprint('\n@ setting up', 'green')


DIR = os.path.dirname(os.path.abspath(__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## ==============================================================================================
osada.cprint('\n@ CNN', 'green')


# if BACKBONE == 'resnet':
#     backbone = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', pooling=None, classes=1000)
# elif BACKBONE == 'vgg':
#     backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
# else:
#     raise ValueError('BACKBONE must be \'resnet\' or \'vgg\'')

# backbone.trainable = False
# backbone.layers[0].trainable = True


## ==============================================================================================
osada.cprint('\n@ take I/O size', 'green')


IMG_SIZE = [224, 224]
# image = np.zeros((1, *IMG_SIZE, 3))
# image = Image.fromarray(image.astype(np.uint8))
# pred = backbone(image)

OUTPUT_SHAPE = [224, 224, 3]

# OUTPUT_SHAPE = pred.shape[1:]
# print(OUTPUT_SHAPE)


## ==============================================================================================
osada.cprint('\n@ mkdir', 'green')


# now = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
# print(now)

# name = f'{BACKBONE}_{now}'

# try:
#     os.makedirs(f'{DIR}/results/{name}')
#     os.makedirs(f'{DIR}/results/{name}/checkpoints')
# except:
#     pass


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
    ds_test.append([image, ans])


def datagenerator(ds, augment_fn, epochs=1, batch_size=16, shuffle=True, augment=1, break_limit=False, test=False, for_vit=False):

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
    
    if for_vit: osada.cprint(f'@ transform images for ViT', 'green')
    
    if test:
        for _ in range(epochs):
            for i, (img, ans) in enumerate(ds):
                if i >= length:break
                
                image = np.array(Image.open(img)).astype(np.float32)
                image = np.array(augment_fn(image))
                if for_vit: image = image.transpose(2, 0, 1)
                # image = backbone.predict([image], verbose=False)[0]
                
                ans = [float(ans)]
                
                
                yield np.array(image), np.array(ans)
        
    else:
        images, answers = [], []
        for _ in range(epochs):
            for i, (img, ans) in enumerate(ds):
                if break_limit and i >= length:break
                if i and not len(answers)%batch_size:
                    # images = backbone.predict(np.array(images), verbose=False)
                    yield np.array(images), np.array(answers)
                    images, answers = [], []
                    
                
                image = np.array(Image.open(img)).astype(np.float32)
                image = np.array(augment_fn(image))
                
                # img_buf = Image.fromarray(np.uint8(image*255.))
                # img_buf.save(f'{DIR}/tmp/check/{i}.png')
                # print(image)
                # if i > batch_size*4: exit()
                
                if for_vit: image = image.transpose(2, 0, 1)
                
                # image = backbone.predict([image], verbose=False)
                
                ans = [float(ans)]
                
                images.append(image)
                answers.append(ans)
                
                # del image, ans
        

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
    # {'batch_size' : 32, 'epochs' : 2, 'mode' : 'min', 'load_weights' : False, 'load_id' : 'previous', 'backbone' : 'resnet'},
    
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'min', 'load_weights' : False, 'load_id' : '', 'backbone' : 'resnet'},
    
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'min', 'load_weights' : False, 'load_id' : '', 'backbone' : 'vit'},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'min', 'load_weights' : True, 'load_id' : 'previous', 'backbone' : 'vit'},
    # {'batch_size' : 32, 'epochs' : 100, 'mode' : 'min', 'load_weights' : True, 'load_id' : 'previous', 'backbone' : 'vit'},
    
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': False, 'load_id': '', 'backbone': 'vgg', 'backbone_trainable': True},
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vgg', 'backbone_trainable': True},
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vgg', 'backbone_trainable': True},
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vgg', 'backbone_trainable': True},
    
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': False, 'load_id': '', 'backbone': 'vgg', 'backbone_trainable': False},
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vgg', 'backbone_trainable': False},
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vgg', 'backbone_trainable': False},
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vgg', 'backbone_trainable': False},
    
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': False, 'load_id': '', 'backbone': 'resnet', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'resnet', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'resnet', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'resnet', 'backbone_trainable': False},

    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'resnet', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'resnet', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'resnet', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'resnet', 'backbone_trainable': False},
    
    
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': False, 'load_id': '', 'backbone': 'vit', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vit', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vit', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vit', 'backbone_trainable': False},

    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vit', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vit', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vit', 'backbone_trainable': False},
    {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vit', 'backbone_trainable': False},
    
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': False, 'load_id': '', 'backbone': 'resnet', 'backbone_trainable': True},
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'resnet', 'backbone_trainable': True},
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'resnet', 'backbone_trainable': True},
    
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': False, 'load_id': '', 'backbone': 'vit', 'backbone_trainable': True},
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vit', 'backbone_trainable': True},
    # {'batch_size': 32, 'epochs': 100, 'mode': 'mid', 'load_weights': True, 'load_id': 'previous', 'backbone': 'vit', 'backbone_trainable': True},
]

# previous = 'resnet_20230628_0451_51'
# buf = 'refine_'
previous = None
buf = ''

# BATCH_SIZE = 32
# EPOCHS = 50
# MODE = 'min'

# BACKBONE = 'resnet'
# BACKBONE_TRAINABLE = True

META_CONFIG = {
    'layernorm' : False,
    'batchnorm' : True,
    'dropout'   : True,
    }

# LOAD_WEIGHTS = False
# LOAD_ID = '20230626_0322_10'

AUGMENT = 2
LR = 1e-7


def my_metric_fn(y_true, y_pred):
    return y_pred


for elem in configs:
    
    key, val = elem.keys(), elem.values()
    BATCH_SIZE, EPOCHS, MODE, LOAD_WEIGHTS, LOAD_ID, BACKBONE, BACKBONE_TRAINABLE = val
    
    if BACKBONE == 'resnet':
        AUGMENT = 4
    if BACKBONE == 'vit':
        AUGMENT = 4
        OUTPUT_SHAPE = [3, 224, 224]
        LR = 1e-4
    
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
    print(now)

    name = f'cnntrain_{BACKBONE}_{"fine-tuning" if BACKBONE_TRAINABLE else "transfer"}_{buf}{now}'

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
    else:
        ep_buf = 0

    with open(f'{DIR}/results/{name}/config.txt', 'w') as f:
        print(f'''\
batch size  : {BATCH_SIZE}
epochs      : {EPOCHS+ep_buf}
mode        : {MODE}
meta_config : {",".join([f"{key}={val}" for key, val in META_CONFIG.items()])}
load_path   : {LOAD_ID if LOAD_WEIGHTS else 'none'}
model       : {BACKBONE}\
''',
file=f)


    train_aug = tf.keras.Sequential([
        layers.Resizing(*IMG_SIZE),
        # layers.Rescaling(1./255.),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        # layers.RandomBrightness(0.2),
    ])
    test_aug = tf.keras.Sequential([
        layers.Resizing(*IMG_SIZE),
        # layers.Rescaling(1./255.),
    ])

    train_datagen = datagenerator(ds_train, train_aug, epochs=EPOCHS,
                                batch_size=BATCH_SIZE, augment=AUGMENT, for_vit=BACKBONE=='vit')
    # test_datagen = datagenerator(ds_test, clip_instance, test_aug, epochs=EPOCHS, batch_size=16, test=True)


    opt = tf.keras.optimizers.Adam(learning_rate=LR)

    model_builder = est_model()
    model = model_builder.build_model(mode=MODE, output_shape=OUTPUT_SHAPE, meta_config=META_CONFIG,
                                    batch_size=BATCH_SIZE, backbone=BACKBONE, backbone_trainable=BACKBONE_TRAINABLE)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])#, run_eagerly=False)
    model.summary()

    if LOAD_WEIGHTS:
        osada.cprint(f'\n@ load weights : {LOAD_ID}', 'orange')
        model.load_weights(f'{DIR}/results/{LOAD_ID}/weights.h5')

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
    # model.save(f'{DIR}/results/{name}')
    model.save_weights(f'{DIR}/results/{name}/weights.h5')

    history = model.history.history
    with open(f'{DIR}/results/{name}/history.txt', 'w') as f:
        print(history, file=f)

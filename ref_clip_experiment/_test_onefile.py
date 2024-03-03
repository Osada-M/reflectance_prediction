import sys
sys.path.append('owl_vit/big_vision/')

import os
from os.path import expanduser

import jax
import numpy as np
import skimage
from skimage import io as skimage_io
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.client import device_lib

from owl_vit import models
from owl_vit.configs import clip_b32
import osada


## ==============================================================================================
osada.cprint('\n@ setting up', 'green')


DIR = f'{expanduser("~")}/workspace/Ritsumei/Laboratory/work/owl_vit_experiment'

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel("ERROR")


## ==============================================================================================
osada.cprint('\n@ OWL-ViT', 'green')


class owlvit:

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        
        self.config_ = clip_b32.get_config(init_mode='canonical_checkpoint')

        self.module = models.TextZeroShotDetectionModule(
            body_configs=self.config_.model.body,
            normalize=self.config_.model.normalize,
            box_bias=self.config_.model.box_bias)
        self.variables_ = self.module.load_variables(self.config_.init_from.checkpoint_path)
    
    
    def run(self, image, text_queries=['keyborad', 'speaker', 'rubik\'s cube', 'pen']):
        
        self.tokenized_queries = np.array([
            self.module.tokenize(q, self.config_.dataset_configs.max_query_length)
            for q in text_queries
        ])
        
        jitted = jax.jit(self.module.apply, static_argnames=('train',))
        
        input_image = skimage.transform.resize(
            image
            (self.config_.dataset_configs.input_size, self.config_.dataset_configs.input_size))

        # Note: The model expects a batch dimension.
        predictions = jitted(
            self.variables_,
            input_image[None, ...],
            self.tokenized_queries[None, ...],
            train=False)

        # Remove batch dimension and convert to numpy:
        predictions = jax.tree_util.tree_map(lambda x: np.array(x[0]), predictions)
        
        return predictions
    
    
    # \n@tf.function
    def __call__(self, image):
        
        token = np.zeros((100, self.config_.dataset_configs.max_query_length), dtype=np.uint32)
        
        jitted = jax.jit(self.module.apply, static_argnames=('train',))

        # Note: The model expects a batch dimension.
        predictions = jitted(
            self.variables_,
            image[None, ...],
            token[None, ...],
            train=False)

        # Remove batch dimension and convert to numpy:
        predictions = jax.tree_util.tree_map(lambda x: np.array(x[0]), predictions)
        
        return np.array(predictions['feature_map'], dtype=np.float32)
        

owl_vit = owlvit()


## ==============================================================================================
osada.cprint('\n@ take I/O size', 'green')


IMG_SIZE = owl_vit.config_.dataset_configs.input_size
print(IMG_SIZE)

image = np.zeros((IMG_SIZE, IMG_SIZE, 3))
pred = owl_vit(image)

OUTPUT_SHAPE = pred.shape
print(OUTPUT_SHAPE)
        

## ==============================================================================================
osada.cprint('\n@ build model', 'green')


def build_model(mode='min'):
    
    inp = tf.keras.layers.Input(shape=OUTPUT_SHAPE)
    x = tf.keras.layers.GlobalAveragePooling2D()(inp)
    
    if mode == 'min':
        pass
    
    elif mode == 'mid':
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    elif mode == 'max':
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='linear')(x)
        
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inp, out)
    
    return model


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


MODE = 'min'
NAME = '20230514_0340_32'
image = f'{DIR}/'

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255),
])

model = build_model(MODE)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()

model.load_weights(f'{DIR}/results/{NAME}/weights.h5')

image = skimage_io.imread(iamge).astype(np.float32)
image = np.array(resize_and_rescale(image))
image = owl_vit(image)

pred = model.predict([image], verbose=0)
    
print(pred)

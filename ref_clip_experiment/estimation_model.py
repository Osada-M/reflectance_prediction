import tensorflow as tf
import numpy as np
from PIL import Image
# from tensorflow.keras import layers
# from tensorflow.keras.applications import EfficientNetB0
from transformers import ViTFeatureExtractor, TFAutoModelForImageClassification, TFAutoModel

from CLIP import CLIP


class Estimation_Model:
    
    def __init__(self):
        
        pass
    
    
    def get_condig(self): return {}
    
    
    # @staticmethod
    ## default : dropout_rate = 0.01
    def build_model(self, mode='min', dropout_rate=0.1, output_shape=(1, 32),
                meta_config={'layernorm':False, 'batchnorm':True, 'dropout':True},
                word_length=10, batch_size=32, backbone=None, backbone_trainable=False,
                verbose=True):
    
        inp = tf.keras.layers.Input(shape=output_shape)
        x = tf.keras.activations.linear(inp)
        
        if not mode.count(' '):
            volume = mode
        else:
            mode, volume = mode.split(' ')
        
        # if mode == 'clip':
            
        #     prompts = [
        #         tf.keras.layers.Dense(32, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=i), activation='sigmoid')(
        #             tf.constant([[1] for _ in range(batch_size)], dtype=tf.float32)
        #             ) for i in range(word_length)
        #         ]
            
        #     out = CLIP_Layer.gradient_func(prompts, x)
            
        #     model = tf.keras.models.Model(inp, out)
        
        #     return model
        
        
        while 1:
            if backbone is not None:
                
                if backbone == 'resnet':
                    # backbone_model = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', pooling=None, classes=1000)
                    backbone_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', pooling=None, classes=1000)
                
                elif backbone == 'vgg':
                    backbone_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
                
                elif backbone == 'vit':
                                        
                    input_layer = tf.keras.layers.Input(shape=(3, 224, 224))
                    vit = TFAutoModel.from_pretrained("google/vit-base-patch32-224-in21k")
                    vit = vit.layers[0]
                    encoded_features = vit(input_layer)['last_hidden_state']
                    backbone_model = tf.keras.models.Model(inputs=input_layer, outputs=encoded_features)
                
                else:
                    break
                
                backbone_model.trainable = backbone_trainable
                backbone_model.layers[0].trainable = True
                
                if verbose:
                    print(f'backbone : {backbone}, {["freezed", "trainable"][backbone_trainable]} : {backbone_model.layers[0].trainable}')
                
                x = backbone_model(x)
            
            break
        
        
        feature_size = len(x.shape)
        
        if verbose:
            print(f'feature size : {feature_size}')
        
        if feature_size == 4:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif feature_size == 3:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
        if meta_config['layernorm']:
            x = tf.keras.layers.LayerNormalization()(x)
        if meta_config['batchnorm']:
            x = tf.keras.layers.BatchNormalization()(x)
        if meta_config['dropout']:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        if volume == 'min':
            pass
        
        elif volume == 'mid':
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            
        elif volume == 'mid_2':
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            x = tf.keras.layers.Dense(32, activation='relu')(x)
        
        elif volume == 'bottleneck':
            x = tf.keras.layers.Dense(16, activation='relu')(x)
            x = tf.keras.layers.Dense(8, activation='relu')(x)
            x = tf.keras.layers.Dense(16, activation='linear')(x)
            
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.models.Model(inp, out)
        
        return model


class CLIP_Layer(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(CLIP_Layer, self).__init__(**kwargs)
        
        self.clip_instance = CLIP(-1)
        
        # self.__start = 49406
        # self.__end = 49707
        # self.__a = 320
        # self.__z = 345
        
        self.__a = ord('a')
        self.__z = ord('z')
        self.__diff = self.__z - self.__a
        
        
    def build(self, input_shape):
        super(CLIP_Layer, self).build(input_shape) 
    
    
    def call(self, prompts, images, output_prompts=False):
        prompts = np.array(prompts).transpose(1, 2, 0)
        # print(prompts.shape)
        prompts = (prompts - np.min(prompts, axis=2, keepdims=True)) /\
            (np.max(prompts, axis=2, keepdims=True) - np.min(prompts, axis=2, keepdims=True))
        
        prompts *= np.float32(self.__diff)
        prompts += np.float32(self.__a)
        # prompts = np.uint32(prompts)
        
        return prompts
        
        text = [[''.join(map(chr, word)) for word in batch] for batch in prompts]
        if output_prompts:
            for batch in text:
                word = batch[0]
                print(word, end=', ')
            print()
                # for word in batch:
                    # print(''.join(map(chr, word)), end=',')
                # print()
        # exit()
        
        features = [[] for i in range(prompts.shape[0])]
        
        '''
        with tf.GradientTape() as tape:
            # images = images.numpy()
        # tf.compat.v1.disable_eager_execution()
            images = np.array(images)
        # images = images.eval(session=tf.compat.v1.Session())
        # images = self.equal(images).numpy()
        if np.max(images) <= 1:
            images *= 255.
        images = np.uint8(images)
        '''
        

        # if tf.reduce_max(images_tensor) <= 1:
            # images_tensor *= 255.
        # images_tensor = tf.cast(images_tensor, tf.uint8)
        
            # img_numpy = images_tensor[i].numpy()
            
            # img_numpy = np.array(images_tensor[i])
            # if np.max(img_numpy) <= 1:
                # img_numpy *= 255.
            # img_numpy = np.uint8(img_numpy)
            # img = Image.fromarray(img_numpy)
            # img = Image.fromarray(images_tensor[i].numpy())
        
        images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
        images = np.array(images_tensor)
        
        for i in range(prompts.shape[0]):
            # img = tf.image.convert_image_dtype(images[i], tf.float32)
            # img = tf.keras.utils.img_to_array(images_tensor[i])
            img = Image.fromarray(images[i])
            print(img)
            print(text[i])
            inputs = self.clip_instance.processor(text=text[i], images=img, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            features[i] = probs.cpu().numpy()
        
        features = np.array(features)
        return features


    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]

    def get_config(self):
        base_config = super(CLIP_Layer, self).get_config()
        return dict(list(base_config.items()))


    @tf.custom_gradient
    def gradient_func(prompts, images):
        y = CLIP_Layer()(prompts, images)
        
        def grad(dy):
            return dy
        
        return y, grad



class ViTProcessingLayer(tf.keras.layers.Layer):
    def __init__(self, feature_extractor, **kwargs):
        super(ViTProcessingLayer, self).__init__(**kwargs)
        self.feature_extractor = feature_extractor

    def call(self, images):
        return {"pixel_values": self.feature_extractor(images)['pixel_values']}

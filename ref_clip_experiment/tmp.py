from transformers import ViTFeatureExtractor, TFAutoModelForImageClassification, TFAutoModel
import numpy as np
import tensorflow as tf


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = TFAutoModelForImageClassification.from_pretrained("google/vit-base-patch32-224-in21k")

vit_model.summary()

# ダミー画像を作成します。実際には、実際の画像データを使用します。
dummy_images = np.ones((1, 224, 224, 3), dtype=np.float32)
dummy_images = np.uint8(dummy_images * 255.0)

preprocessed_images = feature_extractor(dummy_images)['pixel_values']
preprocessed_images = np.expand_dims(preprocessed_images, axis=0)[0]

# print(preprocessed_images)

# モデルに前処理済みの画像を入力します。
# outputs = vit_model(preprocessed_images)
# print(outputs)  # これはクラスのロジットを返します。

input_layer = tf.keras.layers.Input(shape=(3, 224, 224))
vit = TFAutoModel.from_pretrained("google/vit-base-patch32-224-in21k")
vit = vit.layers[0]
encoded_features = vit(input_layer)['last_hidden_state']
backbone_model = tf.keras.models.Model(inputs=input_layer, outputs=encoded_features)

output = backbone_model(preprocessed_images)
print(output)  # これはクラスのロジットを返します。
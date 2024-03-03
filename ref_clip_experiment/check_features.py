import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import clip

# from CLIP import CLIP

DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# clip_instance = CLIP(32)
length = 10

model, preprocess = clip.load("ViT-B/32", device=DEVICE)

img = np.random.random((224, 224, 3)) * 255
img = Image.fromarray(img.astype(np.uint8))
img = preprocess(img).unsqueeze(0).to(DEVICE)
txt = clip.tokenize(['']).to(DEVICE)

logits_per_image, logits_per_text = model(img, txt)
features = logits_per_image.softmax(dim=-1)
print(features.size())

exit()

# for i in range(2):
    
#     if i: clip_instance.prompts = ['' for _ in range(32)]
    
#     for j in range(length):
        
#         image = np.random.random((256, 256, 3)) * 255
#         image = Image.fromarray(image.astype(np.uint8))
#         features = clip_instance(image, use_text=True)
#         features = np.array(features)
        
#         plt.scatter(range(32), features, c='blue')

#     plt.savefig(f'{DIR}/tmp/clip_vit-b32_{i}.png')



for p in map(list, ['az']):
    
    image = np.random.random((256, 256, 3)) * 255
    image = Image.fromarray(image.astype(np.uint8))

    clip_instance.prompts = p.copy()
    inputs = clip_instance.processor(text=clip_instance.prompts, images=image, return_tensors="pt", padding=True)
    image_1 = np.array(inputs['pixel_values'])

    # print(inputs['input_ids'])
    print(inputs)
    print(image_1.shape)

# clip_instance.prompts = ['c', 'd']
# inputs = clip_instance.processor(text=clip_instance.prompts, images=image, return_tensors="pt", padding=True)
# image_2 = np.array(inputs['pixel_values'])

# print(inputs['input_ids'])

# print(np.sum(np.abs(image_1 - image_2)))

# features = clip_instance(image, use_text=True)
# features = np.array(features)
# print(features)
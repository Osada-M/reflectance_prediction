import torch 
import clip
from PIL import Image
import random
import numpy as np
from typing import List
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from cliponnx.models import TextualModel, VisualModel
import matplotlib.pyplot as plt


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os
DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = f'{DIR}/vision-text-map'


class CLIP:

    def __init__(self, prompt_length:int=128, backbone:str='vit', device:str='cuda', use_text:bool=False):
        
        self.backbone = backbone
        self.device = device
        self.use_text = use_text
        
        if self.backbone == 'vit':
            
            # self.vision_model, self.preprocess = clip.load("ViT-B/32", device=DEVICE)
            self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
            self.clip = self.clip.to(self.device)
            self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            self.vision_model = self.clip.vision_model.to(self.device)
            self.text_model = self.clip.text_model.to(self.device)
            
            self.image_to_tensor = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            
            
        elif self.backbone == 'resnet':
            
            ## https://huggingface.co/mlunar/clip-variants
            ## https://huggingface.co/mlunar/clip-variants/blob/main/example.py
            
            providers = ['CPUExecutionProvider']
            self.visual = VisualModel("models/clip-resnet-101-visual-float16.onnx", providers=providers)
            # self.visual = VisualModel(f"{DIR}/models/clip-vit-base-patch32-visual-float16.onnx", providers=providers)
            # self.textual = TextualModel("models/clip-vit-base-patch32-textual-float16.onnx", providers=providers)

        ## default prompt (do not use prediction)
        # if prompt_length == -1:
        self.prompts = ['']
        # else:
        #     length = 10
        #     random.seed(prompt_length)
        #     self.prompts = ['0'*length for _ in range(prompt_length)]
            
        
    def __call__(self, image, text:str=None, save_name:str='default'):
        
        
        if self.backbone == 'vit':
            
            with torch.no_grad():
                
                if self.use_text:
                    
                    if text is None: raise ValueError('text is required')
                    
                    # outputs = self.clip(**inputs)
                    
                    ## vision
                    # vision_tensor = self.image_to_tensor(image).unsqueeze(0).to(self.device)
                    # vision_features = self.vision_model(vision_tensor)['pooler_output']
                    
                    ## text
                    # inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
                    # inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                    # proccessed_text = inputs['input_ids']
                    # text_features = self.text_model(proccessed_text)['pooler_output']
                    
                    # with open(f'{SAVE_DIR}/{save_name}_vision.txt', 'w') as f:
                    #     print(self.vision_model, file=f)
                    # with open(f'{SAVE_DIR}/{save_name}_text.txt', 'w') as f:
                    #     print(self.text_model, file=f)
                    
                    # for save_idx, (vis, txt) in enumerate(zip(vision_features, text_features)):
                    #     vis, txt = vis.cpu().numpy(), txt.cpu().numpy()
                    #     # map = np.outer(vis, txt)
                    #     map = np.vstack([vis, txt])
                    #     correlation_map = np.corrcoef(map)
                    #     print(map.shape)
                    #     plt.imshow(map, cmap='coolwarm', interpolation='nearest')
                    #     plt.colorbar()
                    #     plt.xticks(ticks=np.arange(len(correlation_map)), labels=['vision', 'text'])
                    #     plt.yticks(ticks=np.arange(len(correlation_map)), labels=['vision', 'text'])
                    #     plt.title('Correlation Matrix')
                    #     plt.savefig(f'{SAVE_DIR}/{save_name}_{save_idx:04d}.png')
                    
                    # exit()
                    
                    image = self.image_to_tensor(image).unsqueeze(0).to(self.device)
                    inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
                    inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                    outputs = self.clip(**inputs)
                    
                    text_features = outputs.text_embeds
                    vision_features = outputs.image_embeds                    
                    
                    # features = torch.cat([vision_features, text_features], dim=1)
                    features = (vision_features + text_features) / 2
                                        

                else:
                    # img = self.processor(image).unsqueeze(0).to(self.device)
                    # features = self.vision_model.encode_image(img)
                    
                    vision_tensor = self.image_to_tensor(image).unsqueeze(0).to(self.device)
                    features = self.vision_model(vision_tensor)['last_hidden_state']
        
        elif self.backbone == 'resnet':
            
            images_input = self.visual.preprocess_images(image)
            features = self.visual.encode(images_input)
        
            if self.use_text:
                
                pass
                
        return features
    
    
    def contrast(self, *, image:str=None, prompt:List[str]=None):
        
        return 
        
        image = np.array(Image.open(image))
        image = Image.fromarray(image.astype(np.uint8))
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        text = clip.tokenize(prompt).to(self.device)

        with torch.no_grad():
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = self.vision_model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            return probs


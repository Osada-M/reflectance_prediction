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
import seaborn as sns


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import datetime

NOW = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')

import os
DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = f'{DIR}/vision-text-map/{NOW}'


make_dir = lambda x: os.makedirs(x, exist_ok=True)


class CLIP:

    def __init__(self, prompt_length:int=128, backbone:str='vit', device:str='cuda', use_text:bool=False, is_save_feature:bool=False):
        
        self.backbone = backbone
        self.device = device
        self.use_text = use_text
        self.is_save_feature = is_save_feature
        
        self.savefig_on_random = False
        
        if is_save_feature:
            make_dir(SAVE_DIR)
            with open(f'{SAVE_DIR}/correlation.txt', 'w'): pass
        
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
                    
                    image = self.image_to_tensor(image).unsqueeze(0).to(self.device)
                    inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
                    inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                    outputs = self.clip(**inputs)
                    
                    text_features = outputs.text_embeds
                    vision_features = outputs.image_embeds               
                    
                    ## =====================================================================
                    if self.is_save_feature:
                        
                        txt = text_features.cpu().detach().numpy()[0]
                        vis = vision_features.cpu().detach().numpy()[0]
                        
                        correlation = np.corrcoef(txt, vis)[0][1]
                        
                        with open(f'{SAVE_DIR}/correlation.txt', 'a') as f:
                            print(correlation, file=f)
                        
                        if not self.savefig_on_random or random.random() < 0.05:
                            self.savefig_on_random = True

                            plt.figure(figsize=(10, 10))
                            
                            # map = np.outer(txt, vis)
                            # plt.imshow(map, cmap='coolwarm')
                            
                            plt.scatter(txt, vis, c='r', s=10)
                            plt.xlim(-0.2, 0.2)
                            plt.ylim(-0.2, 0.2)
                            plt.title(f'correlation: {correlation}')
                                                        
                            plt.xlabel('text')
                            plt.ylabel('vision')
                            # plt.savefig(f'{SAVE_DIR}/outer_{save_name}.png')
                            plt.savefig(f'{SAVE_DIR}/scatter_{save_name}.png')
                        
                        # sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
                        # plt.savefig(f'{SAVE_DIR}/{save_name}_correlation.png')
                        
                        
                        # with open(f'{SAVE_DIR}/{save_name}_text.txt', 'w') as f:
                        #     print(txt.shape)
                        #     print(txt, file=f)
                            
                        # with open(f'{SAVE_DIR}/{save_name}_vision.txt', 'w') as f:
                        #     print(vis, file=f)
                            
                        # exit()
                    ## =====================================================================

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
    
    
    def get_correlation(self):
        
        with open(f'{SAVE_DIR}/correlation.txt', 'r') as f:
            lines = f.readlines()
        
        vals = np.asarray([float(x.rstrip('\n')) for x in lines], dtype=np.float32)
        
        mean, std = np.mean(vals), np.std(vals)
        
        return vals, (mean, std)
    
    
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


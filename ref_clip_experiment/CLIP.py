import torch 
import clip
from PIL import Image
import random
import numpy as np
from typing import List
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from cliponnx.models import TextualModel, VisualModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os
DIR = os.path.dirname(os.path.abspath(__file__))


class CLIP:

    def __init__(self, prompt_length:int=128, backbone:str='vit'):
        
        self.backbone = backbone
        
        if self.backbone == 'vit':
            
            # self.model, self.preprocess = clip.load("ViT-B/32", device=DEVICE)
            self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
            self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
            self.model = self.clip.vision_model.to(DEVICE)
            
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

        if prompt_length == -1:
            self.prompts = ['']
        else:
            length = 10
            random.seed(prompt_length)
            self.prompts = [''.join([
                chr(random.randint(ord('A'), ord('z'))) for _ in range(length)
                ]) for _ in range(prompt_length)]
            
        
    def __call__(self, image, use_text:bool=False):
        
        
        if self.backbone == 'vit':
            
            with torch.no_grad():
                
                if use_text:
                    
                    inputs = self.processor(text=self.prompts, images=image, return_tensors="pt", padding=True)
                    outputs = self.clip(**inputs)
                    logits_per_image = outputs.logits_per_image
                    features = logits_per_image.softmax(dim=1)

                else:
                    # img = self.processor(image).unsqueeze(0).to(DEVICE)
                    # features = self.model.encode_image(img)
                    
                    tensor = self.image_to_tensor(image).unsqueeze(0).to(DEVICE)
                    features = self.model(tensor)['last_hidden_state']
        
        elif self.backbone == 'resnet':
            
            images_input = self.visual.preprocess_images(image)
            features = self.visual.encode(images_input)
        
            if use_text:
                
                pass
                
                # texts_input = self.textual.tokenize(self.prompts)
                # text_embeddings = self.textual.encode(texts_input)
                
                # table = [["image", "similarity", "text"]]

                # for ii, image in enumerate(images):
                #     image_embedding = image_embeddings[ii]

                #     similarities = []
                #     for ti, text in enumerate(texts):
                #         text_embedding = text_embeddings[ti]
                #         similarity = cosine_similarity(image_embedding, text_embedding)
                #         similarities.append([similarity, ">" * int(similarity * 30), text])

                #     similarities.sort(reverse=True, key=itemgetter(0))
                #     print(image)
                #     print(tabulate(similarities, headers=["similarity", "bar chart", "text"]))
                #     print()
                
        return features
    
    
    def contrast(self, *, image:str=None, prompt:List[str]=None):
        
        return 
        
        image = np.array(Image.open(image))
        image = Image.fromarray(image.astype(np.uint8))
        image = self.preprocess(image).unsqueeze(0).to(DEVICE)
        
        text = clip.tokenize(prompt).to(DEVICE)

        with torch.no_grad():
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            return probs


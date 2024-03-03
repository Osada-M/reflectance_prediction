from operator import itemgetter
import numpy as np
from tabulate import tabulate

from cliponnx.models import TextualModel, VisualModel

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# With GPU (slower startup, faster inference with supported cards)
# providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# CPU only (faster startup, slower inference)
providers = ['CPUExecutionProvider']

images = [
    "flowers.jpg",
    "heavy-industry.jpg",
]

texts = [
    "a close up photo of a cherry blossom",
    "cherry blossom",
    "flowers",
    "plant",
    "processing plant",
    "a large industrial plant with many pipes, walkways and railings",
    "ruhrgebiet",
    "industry",
    "a photo taken on a bright and sunny day",
    "a photo taken on a dark and cloudy day",
    "a photo taken at midnight",
    "bees",
    "cars",
    "dogs and cats",
]

visual = VisualModel("models/clip-vit-base-patch32-visual-float16.onnx", providers=providers)
images_input = visual.preprocess_images(images)
print(f"Images shape: {images_input.shape}")
image_embeddings = visual.encode(images_input)
print(f"Embeddings shape: {image_embeddings.shape}")
print()

textual = TextualModel("models/clip-vit-base-patch32-textual-float16.onnx", providers=providers)
texts_input = textual.tokenize(texts)
print(f"Texts shape: {texts_input.shape}")
text_embeddings = textual.encode(texts_input)
print(f"Embeddings shape: {text_embeddings.shape}")
print()

table = [["image", "similarity", "text"]]

for ii, image in enumerate(images):
    image_embedding = image_embeddings[ii]

    similarities = []
    for ti, text in enumerate(texts):
        text_embedding = text_embeddings[ti]
        similarity = cosine_similarity(image_embedding, text_embedding)
        similarities.append([similarity, ">" * int(similarity * 30), text])

    similarities.sort(reverse=True, key=itemgetter(0))
    print(image)
    print(tabulate(similarities, headers=["similarity", "bar chart", "text"]))
    print()

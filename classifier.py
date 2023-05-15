#This file takes in a question and the clip results, and fetches the most plausible action 

import langchain
import os
from api import *
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests

class LLM:
    def __init__(self, model):
        if model == "clip":
            self.model = None
        else: 
            self.model = None

    
    def classify(self):
        imagenette = load_dataset(
            'frgfm/imagenette',
            '320px',
            split='validation',
            revision="4d512db"
        )
        # show dataset info
        print(imagenette)
        labels = imagenette.info.features['label'].names
        clip_labels = [f"a photo of a {label}" for label in labels]
        model_id = "openai/clip-vit-base-patch32"

        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)

        # if you have CUDA set it to the active device like this
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # move the model to the device
        model.to(device)
        print(device)

        label_tokens = processor(
            text=clip_labels,
            padding=True,
            images=None,
            return_tensors='pt'
        ).to(device)

        # encode tokens to sentence embeddings
        label_emb = model.get_text_features(**label_tokens)

        # detach from pytorch gradient computation
        label_emb = label_emb.detach().cpu().numpy()
        label_emb.shape

        # normalization
        label_emb = label_emb / np.linalg.norm(label_emb, axis=0)
        label_emb.min(), label_emb.max()

        print(imagenette[0]['image'])

        image = processor(
            text=None,
            images=imagenette[0]['image'],
            return_tensors='pt'
        )['pixel_values'].to(device)
        print(image.shape)
        img_emb = model.get_image_features(image)
        img_emb = img_emb.detach().cpu().numpy()
        scores = np.dot(img_emb, label_emb.T)

        # get index of highest score
        pred = np.argmax(scores)

        # find text label with highest score
        print(labels[pred])
    
    def classify2(self):
        model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer       = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        greedy = True
        model = model_raw
        image = Image.open(requests.get(url, stream =True).raw)
        pixel_values   = image_processor(image, return_tensors ="pt").pixel_values
        plt.imshow(np.asarray(image))
        plt.show()

        if greedy:
            generated_ids  = model.generate(pixel_values, max_new_tokens = 30)
        else:
            generated_ids  = model.generate(
                pixel_values,
                do_sample=True,
                max_new_tokens = 30,
                top_k=5)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# url = "https://raw.githubusercontent.com/michael-franke/npNLG/main/neural_pragmatic_nlg/pics/06-3DS-example.jpg"
# url = "https://img.welt.de/img/sport/mobile102025155/9292509877-ci102l-w1024/hrubesch-rummenigge-BM-Berlin-Gijon-jpg.jpg"
# url = "https://faroutmagazine.co.uk/static/uploads/2021/09/The-Cover-Uncovered-The-severity-of-Rage-Against-the-Machines-political-message.jpg"
# url = "https://media.npr.org/assets/img/2022/03/13/2ukraine-stamp_custom-30c6e3889c98487086d76869f8ba6a8bfd2fd5a1.jpg"


if __name__ == "__main__":
    model = LLM(model="clip")
    model.classify2()

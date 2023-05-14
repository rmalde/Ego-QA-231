#This file takes in a question and the clip results, and fetches the most plausible action 

import langchain
import os
from api import *
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

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


if __name__ == "__main__":
    model = LLM(model="clip")
    model.classify()

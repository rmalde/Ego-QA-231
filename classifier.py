#This file takes in a question and the clip results, and fetches the most plausible action 

import langchain
import os
from api import *
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel, MBart50TokenizerFast
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests

url = "https://raw.githubusercontent.com/michael-franke/npNLG/main/neural_pragmatic_nlg/pics/06-3DS-example.jpg"

class Classifier:
    def __init__(self, classifier_name):
        self.classifier_name = classifier_name
        if classifier_name == "nlpconnect/vit-gpt2-image-captioning":
            self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.tokenizer       = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        else: 
            self.model = None
    
    def classify(self, image=None):
        """
            image is a PIL Image
        """
        if image is None:
            #For testing
            image = Image.open(requests.get(url, stream =True).raw)
        
        generated_text = None

        if self.classifier_name == "nlpconnect/vit-gpt2-image-captioning":
            pixel_values = self.image_processor(image, return_tensors ="pt").pixel_values
            generated_ids  = self.model.generate(
                pixel_values,
                do_sample=True,
                max_new_tokens = 30,
                top_k=5)
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            print("Classifier not supported")
            raise RuntimeError
        # Plotting Code
        # plt.imshow(np.asarray(image))
        # plt.show()

        return generated_text


if __name__ == "__main__":
    classifier = Classifier(classifier="nlpconnect/vit-gpt2-image-captioning")
    classifier.classify()
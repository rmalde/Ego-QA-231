# This file takes in a question and the clip results, and fetches the most plausible action

import langchain
import os
import io
from api import *
from datasets import load_dataset
from transformers import (
    CLIPProcessor,
    CLIPModel,
    GPT2TokenizerFast,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    MBart50TokenizerFast,
)
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from promptcap import PromptCap

from params import CaptionerParams


class Captioner:
    def __init__(self, captioner_name, captioner_params):
        self.captioner_name = captioner_name
        self.captioner_params = captioner_params
        if captioner_name == "nlpconnect/vit-gpt2-image-captioning":
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
            self.image_processor = ViTImageProcessor.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
        elif captioner_name == "promptcap":
            self.model = PromptCap("vqascore/promptcap-coco-vqa")
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            raise RuntimeError(f"Unsupported Captioner: {captioner_name}")

    def caption(self, image, question=None, choices=None):
        """
        image is a PIL Image
        """

        generated_text = None

        if self.captioner_name == "nlpconnect/vit-gpt2-image-captioning":
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(
                pixel_values, do_sample=True, max_new_tokens=30, top_k=5
            )
            generated_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        elif self.captioner_name == "promptcap":
            # promptcap needs the image to be a file, not a PIL image
            # TODO: have the dataloader give filenames rather than PIL objects
            file_object = io.BytesIO()
            image.save(file_object, format='PNG')
            file_object.seek(0)

            if self.captioner_params.question_type == CaptionerParams.Configs.Caption or question is None:
                query = "what does the image describe?"
            elif self.captioner_params.question_type == CaptionerParams.Configs.Q_Caption:
                query = (
                    "please describe this image verbosely according to the given question: "
                    + question + "?"
                )
            elif self.captioner_params.question_type == CaptionerParams.Configs.Q:
                query = (
                    question + "?"
                )

            elif self.captioner_params.question_type == CaptionerParams.Configs.Q_Answer:
                query = (
                    question + "? Your choices are: " + ", ".join(choices) + "."
                )
                print("Query", query)

            elif self.captioner_params.question_type == CaptionerParams.Configs.Q_Cracked:
                query = (
                    "A person is trying to ask questions about an image to an AI model, that will then reply with the answer. However, their questions may be unclear in terms of what they are looking for. Given the following question, " + question + "and the following answers the user is expecting, " + ", ".join(choices) + ", come up with a better and more detailed question to ask the AI model"
                )

            generated_text = self.model.caption(
                query, file_object
            ) 
            

        else:
            raise RuntimeError(f"Unsupported Captioner: {self.captioner_name}")

        # Plotting Code
        # plt.imshow(np.asarray(image))
        # plt.show()

        print("Query", query)
        print("Generated Text: ", generated_text)

        return generated_text


if __name__ == "__main__":
    # dummy image
    url = "https://raw.githubusercontent.com/michael-franke/npNLG/main/neural_pragmatic_nlg/pics/06-3DS-example.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # # test ViT
    # print("Testing ViT Captioner...")
    # captioner = Captioner("nlpconnect/vit-gpt2-image-captioning", CaptionerParams)
    # caption = captioner.caption(image)
    # print("Caption: ", caption)

    # test PromptCap
    print("Testing PromptCap")
    captioner = Captioner("promptcap", CaptionerParams)
    caption = captioner.caption(image, question="What is the color of the sky?")
    print("Caption with query: ", caption)
    caption = captioner.caption(image)
    print("Caption no query: ", caption)

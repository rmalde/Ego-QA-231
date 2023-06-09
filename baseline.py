import random
import os
from llm import LLM
from captioner import Captioner
from tqdm.rich import tqdm
import time
import numpy as np

import torchvision.transforms as transforms

from params import MainParams, BaselineParams, CaptionerParams

"""This approach simple takes the middle frame from each questions, 
uses CLIP to caption that immage, and passes that result plus the quesiton into an LLM
to answer the question"""


class Baseline_Test:
    def __init__(
        self, questions, frames, answers, answer_choices, model_name, captioner_name
    ):
        self.questions = questions
        self.frames = frames
        self.answers = answers
        self.answer_choices = answer_choices
        self.model_name = model_name
        self.captioner_name = captioner_name

    def sample(self, num_samples):
        # Randomly get number of indices to sample
        num_samples = min(num_samples, len(self.questions))
        # Randomly sample indices
        random.seed(34)
        indices = random.sample(range(len(self.questions)), num_samples)
        # Return the questions, frames, answers, and answer choices at those indices
        return (
            [self.questions[i] for i in indices],
            [self.frames[i] for i in indices],
            [self.answers[i] for i in indices],
            [self.answer_choices[i] for i in indices],
        )

    def extract_frames(self, frames_tensor, n_caption_frames=1):
        """_summary_

        Args:
            frames_tensor: shape = (n_frames, C, H, W)
            n_caption_frames [int]: how many frames in video to return

        Returns:
            chosen_frames torch.tensor: shape = (n_caption_frames, C, H, W)
        """
        if n_caption_frames == 1:
            # Extract the relevant frames given the frame list
            indices = [int(len(frames_tensor) / 2)]
        else:
            length = len(frames_tensor)
            indices = np.linspace(0, length-1, n_caption_frames + 2, dtype=int)
            indices = indices[1:-1]
            
        return frames_tensor[indices]

    def caption_frames(
        self,
        frames,
        captioner,
        question,
        choices=None,
        verbose=False,
    ):
        """
        Args: 
            frames: torch.tensor: shape = (n_caption_frames, C, H, W)
        
        Returns:
            caption: str
        """
        caption = ""
        for i, frame in enumerate(frames):
            image = transforms.ToPILImage()(frame)
            # Pass frame into CLIP and retrieve result
            caption += f"Timestamp {i+1}: "
            caption += captioner.caption(image, question=question, choices=choices)
            caption += ", "
        if verbose:
            print("Caption: ", caption)
        return caption

    def answer_question(
        self,
        model,
        question,
        answer_choices,
        caption,
        show_choices=True,
        use_clip=True,
        verbose=False,
    ):
        if verbose:
            print(f"Question: {question}?")
        # Pass question and CLIP result to GPT and retrieve answer
        model.initialize(question, caption, answer_choices)
        chosen_answer = model.answer(show_choices, use_clip)
        return chosen_answer

    def evaluate_answer(self, chosen_answer, correct_answer, verbose=False):
        # Evaluate the answer compared to the actual answer
        if verbose:
            print("Chosen answer: ", chosen_answer)
            print("Correct answer: ", correct_answer)
        if correct_answer in chosen_answer.lower():
            if verbose:
                print("Correct!")
            return True
        else:
            if verbose:
                print("Incorrect!")
            return False

    def run_baseline(self, baseline_params, captioner_params):
        """
        baseline_params: dataclass from params.py
        """
        # Run the baseline model

        model = LLM(model_name=self.model_name)
        captioner = Captioner(self.captioner_name, captioner_params)
        (
            sampled_questions,  # ["questions"]
            sampled_frames,  # [ Tensor(n_frames, C, H, W) ]
            sampled_answers,  # ['answers']
            sampled_answer_choices,  # []
        ) = self.sample(baseline_params.num_samples)

        numCorrect = 0
        for i in tqdm(range(len(sampled_questions))):
            frames = self.extract_frames(sampled_frames[i], n_caption_frames=baseline_params.n_caption_frames)
            question = sampled_questions[i]
            caption = self.caption_frames(
                frames,
                captioner,
                question,
                choices=sampled_answer_choices[i],
                verbose=baseline_params.verbose,
            )
            chosen_answer = self.answer_question(
                model,
                sampled_questions[i],
                sampled_answer_choices[i],
                caption,
                baseline_params.show_choices,
                baseline_params.use_clip,
                verbose=baseline_params.verbose,
            )
            result = self.evaluate_answer(
                chosen_answer, sampled_answers[i], verbose=baseline_params.verbose
            )
            time.sleep(np.random.randint(0, 5))

            if result:
                numCorrect += 1

        # Format Results
        accuracy = numCorrect / len(sampled_questions)
        print(
            f"{numCorrect} correct out of {len(sampled_questions)} examples.\nAccuracy: {round(accuracy * 100, 2)}%"
        )


if __name__ == "__main__":
    # Testing code
    from PIL import Image
    import requests

    questions = ["how many people am I talking with"]

    url = "https://raw.githubusercontent.com/michael-franke/npNLG/main/neural_pragmatic_nlg/pics/06-3DS-example.jpg"
    img = Image.open(requests.get(url, stream=True).raw)
    frames = [transforms.ToTensor()(img)[None, :, :, :]]

    answers = ["zero"]
    answer_choices = [["zero", "one", "two", "three", "four"]]

    BaselineParams.verbose = True

    # Create the baseline model
    baseline = Baseline_Test(
        questions,
        frames,
        answers,
        answer_choices,
        MainParams.model_name,
        MainParams.captioner_name,
    )
    # Run the baseline model
    baseline.run_baseline(BaselineParams, CaptionerParams)

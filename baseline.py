import random
import os
from llm import LLM
from classifier import Classifier
from tqdm.rich import tqdm

import torchvision.transforms as transforms


'''This approach simple takes the middle frame from each questions, 
uses CLIP to classify that immage, and passes that result plus the quesiton into an LLM
to answer the question'''

class Baseline_Test():
    def __init__(self, questions, frames, answers, answer_choices, model_name, classifier_name):
        self.questions = questions
        self.frames = frames
        self.answers = answers
        self.answer_choices = answer_choices
        self.model_name = model_name
        self.classifier_name = classifier_name

    def sample(self, num_samples):
        # Randomly get number of indices to sample
        num_samples = min(num_samples, len(self.questions))
        # Randomly sample indices
        random.seed(33)
        indices = random.sample(range(len(self.questions)), num_samples)
        # Return the questions, frames, answers, and answer choices at those indices
        return ([self.questions[i] for i in indices], 
               [self.frames[i] for i in indices],   
               [self.answers[i] for i in indices], 
               [self.answer_choices[i] for i in indices])
    
    def extract_frame(self, frames_tensor):
        #Extract the relevant frames given the frame list
        middle_frame = int(len(frames_tensor)/2)
        return frames_tensor[middle_frame]

    def classify_frame(self, frame, classifier):
        """
            frame (torch.tensor): shape = (C, H, W)
        """
        image = transforms.ToPILImage()(frame)
        #Pass frame into CLIP and retrieve result
        clip_result =  classifier.classify(image=image)
        return clip_result

    def answer_question(self, model, question, answer_choices, clip_result, show_choices=True, use_clip=True, verbose=False):
        if verbose: print(f"Question: {question}?")
        #Pass question and CLIP result to GPT and retrieve answer
        model.initialize(question, clip_result, answer_choices)
        chosen_answer = model.answer(show_choices, use_clip)
        return chosen_answer

    def evaluate_answer(self, chosen_answer, correct_answer, verbose=False):
        # Evaluate the answer compared to the actual answer
        if verbose:
            print("Chosen answer: ", chosen_answer[2:])
            print("Correct answer: ", correct_answer)
        if correct_answer in chosen_answer.lower():
            if verbose: print("Correct!")
            return True
        else:
            if verbose: print("Incorrect!")
            return False

    def run_baseline(self, baseline_params):
        """
            baseline_params: dataclass from params.py
        """
        #Run the baseline model
        
        model = LLM(model_name=self.model_name)
        classifier = Classifier(classifier_name=self.classifier_name)
        sampled_questions, sampled_frames, sampled_answers, sampled_answer_choices = self.sample(baseline_params.num_samples)
        numCorrect = 0
        for i in tqdm(range(len(sampled_questions))):
            frame = self.extract_frame(sampled_frames[i])
            clip_result = self.classify_frame(frame, classifier)
            chosen_answer = self.answer_question(model, sampled_questions[i], sampled_answer_choices[i], clip_result, baseline_params.show_choices, baseline_params.use_clip, verbose=baseline_params.verbose)
            result = self.evaluate_answer(chosen_answer, sampled_answers[i], verbose=baseline_params.verbose)

            if result:
                numCorrect += 1
        
        #Format Results
        accuracy = numCorrect / len(sampled_questions)
        print(f"{numCorrect} correct out of {len(sampled_questions)} examples.\nAccuracy: {round(accuracy * 100, 2)}%")


if __name__ == "__main__":
    # Testing code
    questions = ["how many people am I talking with"]
    frames = [[2]]
    answers = ["two"]
    answer_choices = [["zero", "one", "two", "three", "four"]]
    # Create the baseline model
    baseline = Baseline_Test(questions, frames, answers, answer_choices)
    # Run the baseline model
    baseline.run_baseline()
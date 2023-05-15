import random
import os
from llm import LLM
from classifier import Classifier


'''This approach simple takes the middle frame from each questions, 
uses CLIP to classify that immage, and passes that result plus the quesiton into an LLM
to answer the question'''

class Baseline_Test():
    def __init__(self, questions, frames, answers, answer_choices):
        self.questions = questions
        self.frames = frames
        self.answers = answers
        self.answer_choices = answer_choices

    def sample(self, num_samples):
        # Randomly get number of indices to sample
        num_samples = min(num_samples, len(self.questions))
        # Randomly sample indices
        indices = random.sample(range(len(self.questions)), num_samples)
        # Return the questions, frames, answers, and answer choices at those indices
        return ([self.questions[i] for i in indices], 
               [self.frames[i] for i in indices],   
               [self.answers[i] for i in indices], 
               [self.answer_choices[i] for i in indices])
    
    def extract_frame(self, frame_list=[]):
        #Extract the relevant frames given the frame list
        return None

    def classify_frame(self, frame, model):
        #Pass frame into CLIP and retrieve result
        if frame == None:
            return "dummy result"
        else:
            result = model.classify(frame)

    def answer_question(self, model, question, answer_choices, clip_result):
        #Pass question and CLIP result to GPT and retrieve answer
        model.initialize(question, clip_result, answer_choices)
        chosen_answer = model.answer(show_choices=True)
        return chosen_answer

    def evaluate_answer(self, chosen_answer, correct_answer):
        # Evaluate the answer compared to the actual answer
        print("Chosen answer: ", chosen_answer)
        print("Correct answer: ", correct_answer)
        if correct_answer in chosen_answer:
            print("Correct!")
        else:
            print("Incorrect!")

    def run_baseline(self, model_name="openai", classifier_name="clip"):
        #Run the baseline model
        model = LLM(model=model_name)
        classifier = Classifier(model=classifier_name)
        sampled_questions, sampled_frames, sampled_answers, sampled_answer_choices = self.sample(5)
        for i in range(len(sampled_questions)):
            print(sampled_frames[i])
            frame = self.extract_frame(frame_list=sampled_frames[i])
            clip_result = self.classify_frame(frame)
            chosen_answer = self.answer_question(model, sampled_questions[i], sampled_answer_choices[i], clip_result)
            self.evaluate_answer(chosen_answer, sampled_answers[i])

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
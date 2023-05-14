import random
import langchain
import os

'''This approach simple takes the middle frame from each questions, 
uses CLIP to classify that immage, and passes that result plus the quesiton into an LLM
to answer the question'''

class Baseline_Test():
    def __init__(self, questions, frames, answers, answer_choices):
        self.questions = questions
        self.frames = frames
        self.answers = answers
        self.answer_choices = answer_choices

    def sample(self):
        # Randomly get number of indices to sample
        num_samples = random.randint(0, len(self.questions))
        # Randomly sample indices
        indices = random.sample(range(len(self.questions)), num_samples)
        # Return the questions, frames, answers, and answer choices at those indices
        return self.questions[indices], self.frames[indices], self.answers[indices], self.answer_choices[indices]
    
    def extract_frame(frame_list):
        #Extract the frame given the question and the id 
        return None

    def classify_frame(frame):
        #Pass frame into CLIP and retrieve result
        if frame == None:
            return "dummy result"

    def answer_question(question, answer_choices, clip_result):
        #Pass question and CLIP result to GPT and retrieve answer
        pass

    def evaluate_answer():
        # Evaluate the answer compared to the actual answer
        pass

    def run_baseline(self):
        #Run the baseline model
        sampled_questions, sampled_frames, sampled_answers, sampled_answer_choices = self.sample()
        for i in range(len(sampled_questions)):
            frame = self.extract_frame(sampled_frames[i])
            clip_result = self.classify_frame(frame)
            chosen_answer = self.answer_question(sampled_questions[i], sampled_answer_choices[i], clip_result)
            self.evaluate_answer(chosen_answer, sampled_answers[i])

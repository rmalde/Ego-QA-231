import random

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
    
    def extract_frame():
        #Extract the frame given the question and the id 
        pass

    def classify_frame():
        #Pass frame into CLIP and retrieve result 
        pass

    def answer_question():
        #Pass question and CLIP result to GPT and retrieve answer
        pass

    def evaluate_answer():
        # Evaluate the answer compared to the actual answer
        pass

    def run_baseline():
        #Run the baseline model
        pass
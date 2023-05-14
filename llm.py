#This file takes in a question and the clip results, and fetches the most plausible action 

import langchain
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from api import *

class LLM:
    def __init__(self, model):
        if model == "openai":
            os.environ["OPENAI_API_KEY"] = openai_key
            self.model = OpenAI(model_name="text-davinci-003")
        else: 
            self.model = None
        self.question = None
        self.clip_result = None
        self.answer_choices = None

        self.no_choices = "Given an image about {clip_result}, {question}? Your answer is:"
        self.choices = "Given an image about {clip_result}, {question}? Your answer choices are: {answer_choices}. Output only the answer choice. Your answer is:"
    
    def initialize(self, question, clip_result, answer_choices):
        self.question = question
        self.clip_result = clip_result
        self.answer_choices = answer_choices
    
    def answer(self, show_choices):
        template = None
        if show_choices:
            template = self.choices
        else:
            template = self.no_choices
        
        prompt = PromptTemplate(
        input_variables=["clip_result", "question", "answer_choices"],
        template=template,
        )

        result = prompt.format(clip_result=self.clip_result, 
                               question=self.question, 
                               answer_choices=self.answer_choices)
        
        chain = LLMChain(llm = self.model, 
                         prompt = prompt)
        
        result = chain.run({"clip_result": self.clip_result, "question": self.question, "answer_choices": self.answer_choices})
        return result
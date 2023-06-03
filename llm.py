# This file takes in a question and the clip results, and fetches the most plausible action

import langchain
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from api import *


class LLM:
    def __init__(self, model_name):
        # TODO: Add a "model verbosity" thing that can include an "explain your reasoning" in the prompt
        if model_name == "openai":
            os.environ["OPENAI_API_KEY"] = openai_key
            self.model = OpenAI(model_name="gpt-3.5-turbo")
        else:
            self.model = None
        self.question = None
        self.clip_result = None
        self.answer_choices = None

        # self.no_choices = "Given an image about {clip_result}, {question}? Your answer is:"
        # self.choices = "Given an image about {clip_result}, {question}?Your answer choices are:{answer_choices}.Your answer should only include the answer choice. Your answer is:"

    def initialize(self, question, clip_result, answer_choices):
        self.question = question
        self.clip_result = clip_result
        self.answer_choices = answer_choices

    def build_template(self, show_choices, use_clip):
        template = ""
        input_variables = []
        chain_dict = {}
        if use_clip:
            template += "Given an image about {clip_result}, "
            chain_dict["clip_result"] = self.clip_result
            input_variables.append("clip_result")
        else:
            template += "Answer the following question: "

        template += "{question}? "
        chain_dict["question"] = self.question
        input_variables.append("question")

        if show_choices:
            template += "Your answer choices are:{answer_choices}.Your answer should only include the answer choice. "
            chain_dict["answer_choices"] = self.answer_choices
            input_variables.append("answer_choices")
        else:
            template += "Your answer is:"
        return template, input_variables, chain_dict

    def answer(self, show_choices, use_clip):
        template, input_variables, chain_dict = self.build_template(
            show_choices, use_clip
        )

        prompt = PromptTemplate(
            input_variables=input_variables,
            template=template,
        )

        chain = LLMChain(llm=self.model, prompt=prompt)

        result = chain.run(chain_dict)
        return result


if __name__ == "__main__":
    llm = LLM(model_name="openai")
    question = "how many people am I talking with"
    clip_result = "a dog eating a human"
    answer_choices = ["zero", "one", "two", "three", "four"]
    llm.initialize(question, clip_result, answer_choices)

    # test all the show_choices and use_clip options
    for show_choices in [True, False]:
        for use_clip in [True, False]:
            chosen_answer = llm.answer(show_choices=show_choices, use_clip=use_clip)
            # print(chosen_answer)

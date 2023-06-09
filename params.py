from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class PreprocessParams:
    img_H = 224
    img_W = 224
    img_C = 3
    # Dataset types:
    # baseline:  X = (question, frames)   y = answer
    # choices:  X = (question, frames, choices) y = answer
    dataset_type = "choices"
    dataset_save_path = "data/data_save.pt"


@dataclass(frozen=True)
class MainParams:
    # captioner_name = "nlpconnect/vit-gpt2-image-captioning"
    captioner_name = "promptcap"
    model_name = "openai"


@dataclass(frozen=True)
class BaselineParams:
    use_clip = True
    show_choices = True
    num_samples = 20
    verbose = True
    n_caption_frames = 1


@dataclass(frozen=True)
class CaptionerParams:
    class Configs(Enum):
        Caption = 1
        Q_Caption = 2
        Q = 3
        Q_Answer = 4
        Caption_Q_Answer = 5
        Q_Cracked = 6
        VQA = 7
        VQA_Answer = 8
    question_type = Configs.Q_Answer


'''
Prompt Cap (Caption, Answer), GPT (Answer + Answer choices, guess) 
Prompt Cap (Question+Caption, Answer), GPT (Answer + Answer choices, guess) - 50%
PromptCap (Question, Answer), GPT (Answer + Answer Choices, guess) - 70%
PromptCap (Question + Answer Choices, Answer) 
PromptCap (Cracked Question, Answer), GPT (Answer + Answer Choices, guess) 
'''
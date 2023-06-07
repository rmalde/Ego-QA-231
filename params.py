from dataclasses import dataclass


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
    num_samples = 10  # TODO: This number is not affecting anything
    verbose = True

@dataclass(frozen=True)
class CaptionerParams:
    use_question = True
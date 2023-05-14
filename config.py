from dataclasses import dataclass

@dataclass(frozen=True)
class PreprocessParams:
    img_H = 224
    img_W = 224
    img_C = 3
    # Dataset types:
    # baseline:  X = (question, frames)   y = answer
    # choices:  X = (question, frames, choices) y = answer 
    dataset_type = "baseline"   

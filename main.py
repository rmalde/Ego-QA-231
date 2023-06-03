from preprocess import Preprocessor
from params import PreprocessParams, MainParams, BaselineParams, CaptionerParams
from baseline import Baseline_Test
from tqdm.rich import tqdm

from PIL import Image
import requests

import torchvision.transforms as transforms


def load_dummy_dataset():
    questions = ["how many people am I talking with"]

    url = "https://raw.githubusercontent.com/michael-franke/npNLG/main/neural_pragmatic_nlg/pics/06-3DS-example.jpg"
    img = Image.open(requests.get(url, stream=True).raw)
    frames = [transforms.ToTensor()(img)[None, :, :, :]]

    answers = ["zero"]
    answer_choices = [["zero", "one", "two", "three", "four"]]
    return questions, frames, answers, answer_choices


def load_real_dataset():
    data_json_path = "data/qa_data.json"
    frames_dir_path = "data/frames/"

    preprocessor = Preprocessor(data_json_path, frames_dir_path, PreprocessParams)
    questions, frames, answers, answer_choices = preprocessor.create_dataset()
    return questions, frames, answers, answer_choices

def load_one_frame_real_dataset():
    img_path = "data/frames/3/PNGImages/M/00004.png"
    img = Image.open(img_path)
    frames = [transforms.ToTensor()(img)[None, :, :, :]]

    questions = ["how many people am I talking with"]
    answers = ["zero"]
    answer_choices = [["zero", "one", "two", "three", "four"]]
    return questions, frames, answers, answer_choices


if __name__ == "__main__":
    # questions, frames, answers, answer_choices = load_one_frame_real_dataset()
    questions, frames, answers, answer_choices = load_real_dataset()
    print("dataset created")

    baseline_params = BaselineParams
    captioner_params = CaptionerParams

    baseline = Baseline_Test(
        questions,
        frames,
        answers,
        answer_choices,
        model_name=MainParams.model_name,
        captioner_name=MainParams.captioner_name,
    )

    baseline.run_baseline(baseline_params, captioner_params)

    # for show_choices in [True, False]:
    #     for use_clip in [True, False]:
    #         print(f"Use clip: {use_clip}, show choices: {show_choices}")
    #         baseline_params.use_clip = use_clip
    #         baseline_params.show_choices = show_choices
    #         baseline.run_baseline(baseline_params, captioner_params)
    #         print("----------------------------------------")

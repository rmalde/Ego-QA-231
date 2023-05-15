from preprocess import Preprocessor
from params import PreprocessParams, MainParams, BaselineParams
from baseline import Baseline_Test
from tqdm.rich import tqdm

data_json_path = 'data/qa_data.json'
frames_dir_path = 'data/frames/'

preprocessor = Preprocessor(data_json_path, frames_dir_path, PreprocessParams)
questions, frames, answers, answer_choices = preprocessor.create_dataset()
print("dataset created")

baseline_params = BaselineParams

baseline = Baseline_Test(
    questions, frames, answers, answer_choices,
    model_name=MainParams.model_name,
    classifier_name=MainParams.classifier_name
)

baseline_params.num_samples = 570

for show_choices in [True, False]:
        for use_clip in [True, False]:
            print(f"Use clip is{use_clip} and show choices is {show_choices}")
            baseline_params.use_clip = use_clip
            baseline_params.show_choices = show_choices
            baseline.run_baseline(baseline_params)
            print("----------------------------------------")
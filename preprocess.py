from params import PreprocessParams

import torch
import torchvision.transforms as transforms

from PIL import Image
import json
import os
from tqdm.rich import tqdm


class Preprocessor:
    def __init__(self, data_json_path, frames_dir_path, preprocess_params):
        self.data_json_path = data_json_path
        self.frames_dir_path = frames_dir_path
        self.preprocess_params = preprocess_params

    def single_frame_to_tensor(self, img_path):
        image = Image.open(img_path)

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.preprocess_params.img_H, self.preprocess_params.img_H)
                ),
                transforms.ToTensor(),
            ]
        )

        tensor = transform(image)
        return tensor

    def filenames_to_frames_tensor(self, filenames):
        # (Frames, C, H, W)
        tensor = torch.zeros(
            len(filenames),
            self.preprocess_params.img_C,
            self.preprocess_params.img_H,
            self.preprocess_params.img_H,
        )

        for i, filename in enumerate(filenames):
            tensor[i] = self.single_frame_to_tensor(filename)
        return tensor

    def datapoint_to_filenames(self, datapoint):
        filenames = []
        start, end, video, cam = (
            datapoint["start"],
            datapoint["end"],
            datapoint["video"],
            datapoint["cam"],
        )
        for i in range(datapoint["start"], datapoint["end"] + 1):
            file_num = "{:0>5}".format(i)  # converts 42 into 00042
            path = os.path.join(
                self.frames_dir_path, str(video), "PNGImages", cam, file_num + ".png"
            )
            if not os.path.exists(path):
                raise FileNotFoundError
            filenames.append(path)
        return filenames

    def create_dataset(self):
        if os.path.exists(self.preprocess_params.dataset_save_path):
            print(f"Loading dataset from {self.preprocess_params.dataset_save_path}")
            dataset = torch.load(self.preprocess_params.dataset_save_path)
            return dataset

        print(f"Building dataset of type {self.preprocess_params.dataset_type}...")

        with open(self.data_json_path, "r") as f:
            metadata = json.load(f)

        # metadata = metadata[-1:] # TEMPORARY LINE
        questions = []
        answers = []
        frames = []
        answer_choices = []

        for datapoint in tqdm(
            metadata
        ):  # {video, cam, start,end, question, answer, a1, a2, a3, a4, a5, label, question_encode, video_cam, a1_encoder, a2_encoder, a3_encoder, a4_encoder, a5_encoder}
            # build video frames tensor
            try:
                frame_filenames = self.datapoint_to_filenames(datapoint)
                frames_tensor = self.filenames_to_frames_tensor(frame_filenames)
                frames.append(frames_tensor)
                questions.append(datapoint["question"])
                answers.append(datapoint["answer"])
                if self.preprocess_params.dataset_type == "choices":
                    answer_list = []
                    for i in range(1, 6):
                        try:
                            answer_list.append(datapoint[f"a{i}"])
                        except KeyError:
                            break
                    answer_choices.append(answer_list)
            except FileNotFoundError:  # skip any files that are not in the dataset
                continue

        assert len(frames) == len(questions) == len(answers)

        N = len(metadata)
        print(f"Initial Dataset Length: {N}")
        print(f"Processed Dataset Length: {len(frames)}")
        print(f"Percent used: {round(len(frames) / N * 100)}%")

        if self.preprocess_params.dataset_type == "baseline":
            dataset = (questions, frames, answers)
        if self.preprocess_params.dataset_type == "choices":
            dataset = (questions, frames, answers, answer_choices)

        print(f"Saving dataset to {self.preprocess_params.dataset_save_path}")
        torch.save(dataset, self.preprocess_params.dataset_save_path)
        return dataset


if __name__ == "__main__":
    data_json_path = "data/qa_data.json"
    frames_dir_path = "data/frames/"

    preprocessor = Preprocessor(data_json_path, frames_dir_path, PreprocessParams)
    questions, frames, answers, answer_choices = preprocessor.create_dataset()

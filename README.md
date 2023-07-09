# A Second Pair of Eyes: Composing Foundation Models for Egocentric Video QA
Final Project for CS231N by Ronak Malde and Arjun Karanam.

[Final Paper](https://github.com/rmalde/Ego-QA-231/blob/main/231N%20Final%20Report.pdf) | [Final Poster]([url](https://github.com/rmalde/Ego-QA-231/blob/main/231N%20Final%20Poster.pdf))

### Abstract

In this paper, we seek to tackle the project of Egocentric Video Question and Answer, with the goal of creating Augmented Reality systems that one could query for more information about the world around them. As opposed to traditional techniques of joint training across all modalities (in this case, egocentric video and language), we instead take the approach of composing multiple foundation models, using multi-modal informed captioning. This allows us to leverage the powerful priors in these foundational models while finetuning just one part, the Vision Language Model, with our egocentric data. We find that a pairing of PromptCap (a multimodal Vision Language Model) finetuned on data-augmented Egocentric videos + captions, composed with GPT3 yields the best results on the task set forth by the EgoVQA dataset. Using a separate Vision model to generate captions and GPT3 to answer the questions does not perform as well, demonstrating that there is still merit to jointly training a model with Egocentric Video and QA data in pursuit of the Egocentric Video Question and Answering task.

### Environment

Our testing was done using Python 3.8
To set up the environment, install necessary requirements from `requirements.txt` 

```pip install -r requirements.txt```

### Usage

All tests can be run from `main.py`, by simple running

```python main.py```

Parameters are set in `params.py` to run different tests, edit this file with different parameters to run specific tests. The paper explores changing the parameter `CaptionerParams.question_type`, which can take on any value in `CaptionerParams.Configs` for different prompts for the captioning module. 
The paper also explores the parameter `BaselineParams.n_caption_frames`, which sets how many frames are used to construct the world state. 


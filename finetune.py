from promptcap import PromptCap, OFAModel, OFATokenizer

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms

import io

from tqdm.rich import tqdm



class FinetuneDataset(Dataset):
    def __init__(self, dataset_path):
        questions, frames, summaries  = torch.load(dataset_path)
        assert len(questions) == len(frames) == len(summaries)
        self.questions = questions
        self.frames = frames
        self.summaries = summaries
        
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        # tok_summary = self.tokenizer(self.summaries[index], return_tensors="pt").input_ids
        return self.questions[index], self.frames[index], self.summaries[index]

    

class Finetune_Captioner():
    def __init__(self, captioner_name, dataset_path):
        if captioner_name == "promptcap":
            self.model = PromptCap("vqascore/promptcap-coco-vqa")
            self.tokenizer = self.model.tokenizer
            # self.model = OFAModel.from_pretrained("vqascore/promptcap-coco-vqa", use_cache=True)
            # self.tokenizer = OFATokenizer.from_pretrained("vqascore/promptcap-coco-vqa")
        else:
            raise NotImplementedError(captioner_name)

        self.dataset_path = dataset_path
     
    def print_params(self, mode="trainable"):
        # options = ["trainable", "frozen", "all"]
        for name, param in self.model.named_parameters():
            if mode == "trainable":
                if param.requires_grad:
                    print(name)
            elif mode =="frozen":
                if not param.requires_grad:
                    print(name)
            else:
                print(name)

    def transform_img(self, frame):
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        
        patch_resize_transform = transforms.Compose([
            transforms.Resize((resolution, resolution),
                              transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=mean, std=std)
        ])

        return patch_resize_transform(frame)

    def format_tokens_for_loss(self, tokens):
        """_summary_

        Args:
            tokens: tensor(1, n_tokens)
        """

        # Set maximum sequence length
        max_length = 30
        sequence_length = tokens.shape[1]

        if sequence_length < max_length:
            # Pad the sequence with pad_token
            padding_length = max_length - sequence_length
            padding = torch.full((1, padding_length), 0, dtype=torch.long)
            final_sequence = torch.cat((tokens, padding), dim=1).detach()
            

        elif sequence_length > max_length:
            # Truncate the sequence to max_length
            final_sequence = tokens[:, :max_length]

        else:
            final_sequence = tokens

        return final_sequence.float().flatten()
        


    def finetune(self):
        # Freeze all layers except the last trasnformer block
        trainable_params = []
        for name, param in self.model.named_parameters():
            if not name.startswith('model.encoder.layers.11'):  # Adjust the condition based on your model's architecture
                param.requires_grad = False
            else:
                trainable_params.append(param)
        
        # self.print_params(mode="trainable")

        # Create a TensorDataset
        dataset = FinetuneDataset(dataset_path=self.dataset_path)

        # Create a DataLoader
        # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Define loss function and optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = optim.AdamW(trainable_params, lr=2e-2)

        # Train the model
        epochs = 3

        try:

            for epoch in range(epochs):
                running_loss = 0.0
                
                for question, frame, summary in tqdm(dataset):
                    summary_tok = self.tokenizer(summary, return_tensors="pt").input_ids
                    # prompt_tok = self.tokenizer(question, return_tensors="pt").input_ids

                    #put image in the way they want
                    frame = self.transform_img(frame)
                    # image = frame.unsqueeze(0)

                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    num_beams, no_repeat_ngram_size, max_new_tokens = 5, 3, 50
                    # Forward pass
                    # outputs = self.model.generate(prompt_tok, patch_images=image, 
                    #                     num_beams=num_beams, 
                    #                     no_repeat_ngram_size=no_repeat_ngram_size, 
                    #                     max_new_tokens=max_new_tokens,
                    outputs = self.model.caption(question, frame, use_img_tensor=True, mode="train")
                    outputs = self.format_tokens_for_loss(outputs)
                    summary_tok = self.format_tokens_for_loss(summary_tok)
                    loss = loss_function(outputs, summary_tok)
                    loss = torch.tensor(loss, requires_grad=True)

                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    print(self.model.parameters()[0].grad)
                    # print(trainable_params[0].grad)
                    print(loss.item())
                    
                    running_loss += loss.item()
                
                epoch_loss = running_loss / len(dataset)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        except KeyboardInterrupt:
            print("Saving model weights...")



if __name__ == "__main__":
    dataset_path = "data/finetune_dataset_blip.pt"
    finetune_captioner = Finetune_Captioner("promptcap", dataset_path=dataset_path)
    finetune_captioner.finetune()
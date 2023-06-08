from promptcap import PromptCap

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms

import io



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

    def finetune(self):
        # Freeze all layers except the last trasnformer block
        for name, param in self.model.named_parameters():
            if not name.startswith('model.encoder.layers.11'):  # Adjust the condition based on your model's architecture
                param.requires_grad = False
        
        # self.print_params(mode="trainable")

        # Create a TensorDataset
        dataset = FinetuneDataset(dataset_path=self.dataset_path)

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Define loss function and optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)

        # Train the model
        epochs = 3

        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch in dataloader:
                question, frame, summary = batch

                summary_tok = self.tokenizer(summary, padding=True, truncation=True, max_length=50, return_tensors="pt").input_ids

                #put image in the way they want
                image = transforms.ToPILImage()(frame)
                file_object = io.BytesIO()
                image.save(file_object, format='PNG')
                file_object.seek(0)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(question, file_object)
                print(outputs.shape, summary_tok)
                quit()
                loss = loss_function(outputs, summary_tok)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    dataset_path = "data/finetune_dataset_blip.pt"
    finetune_captioner = Finetune_Captioner("promptcap", dataset_path=dataset_path)
    finetune_captioner.finetune()
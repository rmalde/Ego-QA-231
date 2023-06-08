from promptcap import PromptCap

import torch

class Finetune_Captioner():
    def __init__(self, captioner_name):
        if captioner_name == "promptcap":
            self.model = PromptCap("vqascore/promptcap-coco-vqa")
        else:
            raise NotImplementedError(captioner_name)
     
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
        
        self.print_params(mode="trainable")

        # Convert labels to tensors
        labels = torch.tensor(labels)

        # Create a TensorDataset
        dataset = TensorDataset(input_ids, attention_mask, labels)

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Define loss function and optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)

        # Train the model
        epochs = 3

        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch in dataloader:
                input_ids_batch, attention_mask_batch, labels_batch = batch
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    finetune_captioner = Finetune_Captioner("promptcap")
    finetune_captioner.finetune()
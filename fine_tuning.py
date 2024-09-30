# This script HAS VALIDATION and IS CONTRASTIVE

from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os, torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Floss
from torchvision import transforms

os.chdir("fine-tuning")
merged_files = pd.read_csv("merged_files.csv")
list_image_path = torch.load("final_list_path.pt")  # List of image paths downloaded from create_dataset.py
list_txt = torch.load("final_list_txt.pt")  # List of image titles downloaded from create_dataset.py (from loc.gov API)

EPOCH = 16
BATCH_SIZE = 4
TEMPERATURE = 0.07
LEARNING_RATE = 12e-5
 
# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

# # Load the fine-tuned model checkpoint
# checkpoint = torch.load("fine_tuned_clip_epoch_4.pt", map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])

class AugmentedImageTextDataset(Dataset):
    def __init__(self, list_image_path, list_txt, transform=None):
        self.image_path = list_image_path
        self.texts = list_txt
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        text = self.texts[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, text

def collate_fn(batch):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    texts = [item[1] for item in batch]
    images = [item[0] for item in batch]

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    return batch

# Instantiate the dataset with augmentation

# Data augmentation transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

dataset = AugmentedImageTextDataset(list_image_path, list_txt, transform=transform)

validation_split = 0.2
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, sampler=train_sampler)
valid_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, sampler=valid_sampler)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

best_val_loss = float('inf')
patience = 3  # Number of epochs to wait for improvement before stopping
trigger_times = 0

print("Dataset length:", len(dataset)) 
print("Train DataLoader length:", len(train_dataloader))

# No need to change the model or data loading parts

for epoch in range(EPOCH):
    print(f"Starting epoch {epoch+1}")
    model.train()
    total_loss = 0  # Initialize total loss for the epoch

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCH}", unit="batch"):
        optimizer.zero_grad()

        # Prepare inputs and forward pass
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)

        # Get the logits from the model
        logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text

        # Calculate the symmetric InfoNCE loss
        sim_image_text = logits_per_image / TEMPERATURE
        sim_text_image = logits_per_text / TEMPERATURE

        # Labels are the diagonal of the matrix, where image and text of the same index match
        labels = torch.arange(sim_image_text.size(0), device=device)

        # Calculate the loss for both image-to-text and text-to-image
        loss_img = Floss.cross_entropy(sim_image_text, labels)
        loss_txt = Floss.cross_entropy(sim_text_image, labels)

        # Combine the losses and update total loss
        loss = (loss_img + loss_txt) / 2
        total_loss += loss.item()  # Accumulate the loss
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)  # Calculate average loss
    print(f"Epoch {epoch+1}/{EPOCH}, Training Loss: {avg_train_loss}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

            logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
            sim_image_text = logits_per_image / TEMPERATURE
            sim_text_image = logits_per_text / TEMPERATURE
            labels = torch.arange(sim_image_text.size(0), device=device)
            loss_img = Floss.cross_entropy(sim_image_text, labels)
            loss_txt = Floss.cross_entropy(sim_text_image, labels)
            loss = (loss_img + loss_txt) / 2
            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_dataloader)
    print(f"Epoch {epoch+1}/{EPOCH}, Validation Loss: {avg_val_loss}")

    # Early stopping and checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pt')
        print("Best model saved.")
    else:
        trigger_times += 1
        print(f"Trigger times: {trigger_times}")
        if trigger_times >= patience:
            print("Early stopping!")
            break

    # Save model, optimizer, and training state
    model_save_path = f"fine_tuned_clip_epoch_{epoch+1}_bs8.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    }, model_save_path)
    print(f"Model saved to {model_save_path}")

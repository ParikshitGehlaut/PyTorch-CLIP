import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from PIL import Image
import requests
from clip import CLIP
from io import BytesIO

# Assuming CLIP model definition is available as previously discussed

img_size = 224
patch_size = 16
in_channels = 3
embed_dim = 512
n_heads = 8
n_layers = 12
mlp_ratio = 4
vocab_size = 49408  # Assuming a vocabulary size of 49408

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(f"using device: {device}")

# Define a simple contrastive loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, image_features, text_features):
        # Calculate cosine similarity
        logits = self.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0)) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, transform=None):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        text = item["sentences_raw"]
        image = Image.open(BytesIO(requests.get(item["filepath"]).content)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        input_ids = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)["input_ids"].squeeze(0)
        
        return image, input_ids

def load_transform():
    # Define your transformations here
    # Example:
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add more transformations as needed
    ])

# Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load the dataset
dataset = load_dataset("nlphuji/flickr30k")

# Apply transformations
transform = load_transform()

# Wrap the HuggingFace dataset
wrapped_dataset = HuggingFaceDataset(dataset, tokenizer, transform=transform)
dataloader = DataLoader(wrapped_dataset, batch_size=2, shuffle=True)

# Initialize the CLIP model
model = CLIP(
        img_size,
        patch_size,
        in_channels,
        embed_dim,
        n_heads,
        n_layers,
        mlp_ratio,
        vocab_size,
        device=device,
        dropout=0.1,)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = ContrastiveLoss()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    for images, input_ids in dataloader:
        optimizer.zero_grad()
        image_features, text_features = model(images, input_ids)
        loss = loss_fn(image_features, text_features)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import pandas as pd
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# --- 1. SETUP THE ENCODER (THE EAR) ---
class Stage1Encoder(nn.Module):
    def __init__(self):
        super(Stage1Encoder, self).__init__()
        # Load pre-trained ResNet-18
        self.backbone = models.resnet18(pretrained=True)
        # Modify the first layer to accept the 1-channel (grayscale) data you saved
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Strip the final classification layer to get the 512-dim embedding
        self.encoder = nn.Sequential(*list(self.backbone.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return torch.flatten(x, 1)

# Initialize and set to eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Stage1Encoder().to(device)
model.eval()

# --- 2. DATA LOADING ---
input_folder = 'spectrogram_data_pt/'
all_embeddings = []
file_names = []

print("Starting Feature Extraction...")

with torch.no_grad():
    for filename in os.listdir(input_folder):
        if filename.endswith(".pt"):
            # Load the raw .pt data
            spectrogram = torch.load(os.path.join(input_folder, filename))
            
            # RESIZING & NORMALIZING (Crucial for ResNet)
            # ResNet expects 224x224. We interpolate our high-res data to fit.
            # Shape: [N_Mels, Time] -> [1, 1, 224, 224]
            input_tensor = spectrogram.unsqueeze(0).unsqueeze(0)
            input_tensor = torch.nn.functional.interpolate(input_tensor, size=(224, 224), mode='bilinear')
            
            # Get the 512 numbers
            embedding = model(input_tensor.to(device))
            
            # Store results
            all_embeddings.append(embedding.cpu().numpy().flatten())
            file_names.append(filename.replace(".pt", ""))
            print(f"Encoded: {filename}")

# --- 3. SAVE THE FEATURE TABLE ---
# Create a DataFrame: [Song_Name, Feature_1, Feature_2, ..., Feature_512]
df = pd.DataFrame(all_embeddings)
df.insert(0, "song_name", file_names)

# Save as CSV for your thesis analysis
df.to_csv("music_feature_vectors.csv", index=False)
# Save as NumPy for faster Stage 2 training
np.save("music_features.npy", np.array(all_embeddings))

print(f"\nSuccess! Saved 512-dim vectors for {len(file_names)} songs.")
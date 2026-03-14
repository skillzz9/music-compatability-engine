import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from core.model import SiameseTwin

class HarmonyDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []
        mapping = {'pos': 1.0, 'neg_hard': -1.0, 'neg_easy': -1.0}
        
        for cat, label in mapping.items():
            path = os.path.join(data_root, cat)
            if os.path.exists(path):
                files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npy')]
                for f in files:
                    self.samples.append((f, label))
                print(f"📊 Loaded {len(files)} samples from {cat}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path)
        t1 = torch.from_numpy(data[0]).unsqueeze(0).float()
        t2 = torch.from_numpy(data[1]).unsqueeze(0).float()
        return t1, t2, torch.tensor(label).float()

def run_training_pipeline(data_dir, epochs=15, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    dataset = HarmonyDataset(data_dir)
    if len(dataset) == 0:
        print("❌ Error: No data found in training_dataset. Run Phase 2 first!")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SiameseTwin().to(device)
    
    criterion = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n🚀 Training started! Total Batches: {len(loader)}")
    print("-" * 30)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (t1, t2, label) in enumerate(loader):
            t1, t2, label = t1.to(device), t2.to(device), label.to(device)
            
            optimizer.zero_grad()
            out1, out2 = model(t1, t2)
            
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print batch progress every 5 batches
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx}/{len(loader)} | Current Loss: {loss.item():.4f}")
            
        avg_loss = epoch_loss / len(loader)
        print(f"✨ Epoch [{epoch+1}/{epochs}] Finished | Average Loss: {avg_loss:.4f}")
        print("-" * 30)

    torch.save(model.state_dict(), "harmony_model.pth")
    print("💾 Model saved successfully as harmony_model.pth")
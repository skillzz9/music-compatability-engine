import torch
from torch.utils.data import DataLoader
from model import SiameseTwin, ContrastiveLoss
from dataset import MultiSongDataset

def train():
    # 1. Initialize
    # Tip: Change "cpu" to "mps" if you have an M1/M2/M3 Mac for 10x speed
    device = torch.device("cpu") 
    model = SiameseTwin().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    # 2. Load Data
    train_set = MultiSongDataset('multi_song_tensors')
    # Reduced batch_size to 2 for better stability on MacBook Air RAM
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    
    print(f"Starting training on {len(train_set)} pairs...")

    # 3. Training Loop
    model.train()
    for epoch in range(100):
        total_loss = 0
        
        # FIXED: Added enumerate and parentheses for correct unpacking
        for i, (t1, t2, label) in enumerate(train_loader):
            t1, t2, label = t1.to(device), t2.to(device), label.to(device)
            
            optimizer.zero_grad()
            out1, out2 = model(t1, t2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # FIXED: Moved inside the loop to see real-time progress
            if i % 10 == 0:
                print(f"  Epoch [{epoch+1}/100] | Batch [{i}/{len(train_loader)}] | Loss: {loss.item():.4f}")
            
        # Summary print after each epoch
        avg_loss = total_loss / len(train_loader)
        print(f"==> Epoch [{epoch+1}/100] Complete. Average Loss: {avg_loss:.4f}")

    # 4. Save the "Brain"
    torch.save(model.state_dict(), "mini_snn_model.pth")
    print("Model saved to mini_snn_model.pth")

if __name__ == "__main__":
    train()
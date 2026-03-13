import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseTwin(nn.Module):
    def __init__(self):
        super(SiameseTwin, self).__init__()
        # 1. Vision Layers (CNN)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 2. Reasoning Layers (Linear)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128), 
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward_one(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# This was the missing class!
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate Euclidean Distance: how far apart the two sounds are in the map
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Loss formula:
        # If label is 0 (Similar), minimize distance
        # If label is 1 (Different), maximize distance up to the 'margin'
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss
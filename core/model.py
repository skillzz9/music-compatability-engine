import torch
import torch.nn as nn

class SiameseTwin(nn.Module):
    def __init__(self):
        super(SiameseTwin, self).__init__()
        
        # 1. Convolutional Layers (The "Eyes")
        # We expect input [1, 84, 300]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # Result: [32, 42, 150]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Result: [64, 21, 75]
        )
        
        # 2. Fully Connected Layers (The "Decision")
        # LazyLinear automatically calculates the input features
        self.fc = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        # Passes both loops through the same shared weights
        return self.forward_one(input1), self.forward_one(input2)
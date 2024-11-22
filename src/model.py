import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 7 * 7, 32),
            nn.BatchNorm1d(32, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 8 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        return x 
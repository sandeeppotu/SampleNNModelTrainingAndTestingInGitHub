import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 3 * 3, 60),
            nn.BatchNorm1d(60, track_running_stats=True),
            nn.ReLU()
        )
        
        self.fc2 = nn.Linear(60, 10)
        
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Ensure input has proper batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = self.dropout(self.conv1(x))
        x = self.dropout(self.conv2(x))
        x = self.dropout(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten while preserving batch dimension
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def print_model_parameters(model):
    """Detailed parameter count breakdown"""
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            params = p.numel()
            print(f"{name}: {params:,}")
            total += params
    print(f"\nTotal parameters: {total:,}")
    return total

# Calculate parameters
model = MNISTModel()
total_params = model.count_parameters()
print(f"Total parameters: {total_params:,}")
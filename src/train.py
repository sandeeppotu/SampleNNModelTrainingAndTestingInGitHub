import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os
import random
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train():
    # Set seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(15, fill=0),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=15,
            fill=0
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),
        transforms.ElasticTransform(alpha=50.0, sigma=5.0, fill=0),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test transform (only normalization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Increased batch size for faster training
    batch_size = 128
    
    print("Loading datasets...")
    # Load datasets with respective transforms
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1000,
        num_workers=4,
        pin_memory=True
    )
    
    # Model initialization
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Higher initial learning rate with OneCycleLR
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_loader)
    
    # OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=1,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    print("\nStarting training for 1 epoch with enhanced model...")
    
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Calculate training accuracy
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
        
        running_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}')
            print(f'Loss: {running_loss/50:.4f}')
            print(f'Training Accuracy: {100 * train_correct / train_total:.2f}%')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n')
            running_loss = 0.0
    
    # Evaluation phase
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f'\nFinal Results:')
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%\n')
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join('models', f'mnist_model_{timestamp}_acc{accuracy:.1f}.pth')
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
    print(f"Model saved: {save_path}\n")

if __name__ == "__main__":
    train() 
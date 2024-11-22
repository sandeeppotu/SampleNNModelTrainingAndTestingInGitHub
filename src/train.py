import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from torch.optim.lr_scheduler import OneCycleLR
import os
from datetime import datetime

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(15, fill=0),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Optimized batch size
    batch_size = 128
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                 transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, 
                                transform=test_transform)
    
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
    
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimized learning rate and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=1,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # Training loop
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'\nAccuracy: {accuracy:.2f}%')
    
    # Save model
    if accuracy > 95:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join('models', 
                                f'mnist_model_{timestamp}_acc{accuracy:.1f}.pth')
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved: {save_path}")

if __name__ == "__main__":
    train() 
import torch
import pytest
from src.model import MNISTModel
import os
import glob

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNISTModel()
    assert count_parameters(model) < 100000, "Model has too many parameters"

def test_input_output_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape is incorrect"

def test_model_accuracy():
    from torchvision import datasets, transforms
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    
    # Load the latest model
    model_files = glob.glob('models/*.pth')
    if not model_files:
        pytest.skip("No trained model found. Run training first.")
    
    latest_model = max(model_files, key=os.path.getctime)
    try:
        model.load_state_dict(torch.load(latest_model))
    except Exception as e:
        pytest.fail(f"Failed to load model: {str(e)}")
    
    # Test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy {accuracy:.2f}% is below 80%"
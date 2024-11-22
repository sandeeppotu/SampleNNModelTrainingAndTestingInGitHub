import torch
import pytest
from src.model import MNISTModel
import glob
import os
from torchvision import datasets, transforms
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNISTModel()
    assert count_parameters(model) < 25000, "Model has too many parameters"

def test_input_output_shape():
    model = MNISTModel()
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (1, 10), "Output shape is incorrect"

def test_model_accuracy():
    from torchvision import datasets, transforms
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    
    model_files = glob.glob('models/*.pth')
    if not model_files:
        pytest.skip("No trained model found")
    
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
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
    assert accuracy > 80, f"Model accuracy {accuracy:.2f}% is below 80%"

def test_model_noise_robustness():
    model = MNISTModel()
    model.eval()
    
    # Create a batch of test data instead of using MNIST dataset
    batch_size = 8
    test_input = torch.randn(batch_size, 1, 28, 28)
    
    # Get base prediction
    with torch.no_grad():
        base_output = model(test_input)
        base_pred = base_output.argmax(1)
        
        # Test with different noise levels
        noise_levels = [0.1, 0.2, 0.3]
        consistencies = []
        
        for noise_level in noise_levels:
            # Add noise
            noisy_input = test_input + torch.randn_like(test_input) * noise_level
            # Get prediction
            noisy_output = model(noisy_input)
            noisy_pred = noisy_output.argmax(1)
            # Calculate consistency
            consistency = (noisy_pred == base_pred).float().mean().item()
            consistencies.append(consistency)
    
    avg_consistency = sum(consistencies) / len(consistencies)
    assert avg_consistency > 0.5, f"Model predictions not consistent under noise: {avg_consistency:.2f}"

def test_activation_ranges():
    model = MNISTModel()
    model.eval()
    
    # Use batch size > 1 for BatchNorm
    test_input = torch.randn(8, 1, 28, 28)
    
    activations = []
    def hook(module, input, output):
        activations.append(output.detach())
    
    # Attach hooks to ReLU layers
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.register_forward_hook(hook)
    
    with torch.no_grad():
        model(test_input)
    
    for idx, activation in enumerate(activations):
        mean_activation = activation.mean().item()
        max_activation = activation.max().item()
        
        assert 0 < mean_activation < 2.0, f"Layer {idx} has unusual mean activation: {mean_activation}"
        assert 0 < max_activation < 10.0, f"Layer {idx} has unusual max activation: {max_activation}"

def test_gradient_flow():
    model = MNISTModel()
    model.train()  # Set to train mode for BatchNorm
    criterion = torch.nn.CrossEntropyLoss()
    
    # Use batch size > 1 for BatchNorm
    batch_size = 8
    test_input = torch.randn(batch_size, 1, 28, 28)
    target = torch.randint(0, 10, (batch_size,))  # Random targets
    
    # Forward pass
    output = model(test_input)
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    gradients = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradients.append((name, grad_norm))
            # Verify gradient exists and is not zero or NaN
            assert grad_norm == grad_norm, f"NaN gradient in {name}"
            assert grad_norm > 0, f"Zero gradient in {name}"
            assert grad_norm < 10, f"Gradient explosion in {name}: {grad_norm}"
    
    assert len(gradients) > 0, "No gradients found"
# MNIST Digit Recognition with CI/CD Pipeline

[![ML Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/ml-pipeline.yml)

This project implements a lightweight Deep Neural Network for MNIST digit recognition with a complete CI/CD pipeline using GitHub Actions. The model is optimized to use less than 25,000 parameters while maintaining high accuracy (>95%) on the MNIST dataset.

## Overview
The project demonstrates:
- Efficient CNN architecture design
- Automated model training and testing
- Real-world digit recognition capabilities
- CI/CD implementation with GitHub Actions
- Comprehensive testing framework

## Project Structure
├── .github
│ └── workflows
│ └── ml-pipeline.yml
├── src
│ ├── model.py # Neural network architecture
│ ├── train.py # Training script
│ ├── test_real_data.py # Script for testing with real images
│ └── create_test_images.py # Utility to create test images
├── tests
│ └── test_model.py # Unit tests for model validation
├── requirements.txt
└── README.md


## Features
- Lightweight CNN architecture (<25,000 parameters)
- Automated testing and deployment pipeline
- Real-world image testing capability
- Model validation checks
- Automated model versioning with timestamps

## Model Architecture
- 2 Convolutional layers with batch normalization
- Max pooling layers
- Dropout for regularization
- Fully connected layers
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)

## Key Features
- **Lightweight Architecture**: ~13,498 parameters
- **High Accuracy**: >95% on MNIST test set
- **Fast Training**: 5 epochs
- **Real-world Testing**: Support for custom digit images
- **Automated Pipeline**: GitHub Actions integration
- **Comprehensive Testing**: Unit tests and validation

## Installation
pip install -r requirements.txt

### Training
bash
python3 src/train.py

### Testing
bash
python3 src/create_test_images.py
python3 src/test_real_data.py


## CI/CD Pipeline Details

### Workflow Steps
1. **Environment Setup**
   - Python 3.10
   - PyTorch installation
   - Dependencies installation

2. **Model Training**
   - 5 epochs
   - Batch size: 64
   - Adam optimizer
   - Learning rate: 0.001

3. **Validation Tests**
   - Parameter count verification (<25,000)
   - Input/output shape validation
   - Accuracy threshold check (>80%)
   - Model artifact archiving

### Model Versioning
Models are saved with format:

mnist_model_YYYYMMDD_HHMMSS_accXX.X.pth

## Performance Metrics
- Training Time: ~3-5 minutes
- Model Size: <1MB
- Parameters: 13,498
- Test Accuracy: >95%
- Memory Usage: <100MB

## Testing Framework
- **Unit Tests**: Parameter count, shapes, accuracy
- **Integration Tests**: End-to-end training
- **Real-world Testing**: Custom image prediction

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## Development Guidelines
- Maintain parameter count below 25,000
- Ensure test accuracy above 95%
- Add tests for new features
- Follow PEP 8 style guide
- Update documentation

## Troubleshooting
- **GPU/CPU Compatibility**: Automatically handles device selection
- **Image Loading**: Supports various formats (PNG, JPG)
- **Model Loading**: Handles different PyTorch versions

## Future Improvements
- [ ] Model quantization
- [ ] TensorBoard integration
- [ ] API deployment
- [ ] Mobile optimization
- [ ] Extended data augmentation

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- PyTorch Framework
- GitHub Actions
- Testing Framework: pytest

## Contact
Create an issue for questions or suggestions.

## Citation
```bibtex
@software{mnist_lightweight_cnn,
  author = {sandeeppotu},
  title = {Lightweight MNIST CNN with CI/CD},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/sandeeppotu/SampleNNModelTrainingAndTestingInGitHub}
}
```

[![ML Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/ml-pipeline.yml)



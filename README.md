# MNIST Digit Recognition with CI/CD Pipeline

This project implements a Deep Neural Network for MNIST digit recognition with a complete CI/CD pipeline using GitHub Actions. The model is designed to be lightweight (<25,000 parameters) while maintaining high accuracy.

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

## Requirements
pip install -r requirements.txt

## Local Development

1. Set up virtual environment:

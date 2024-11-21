import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import MNISTModel
import glob
import os
import numpy as np

def preprocess_image(image_path):
    # Open image and convert to grayscale
    image = Image.open(image_path).convert('L')
    
    # Add padding to make it square
    w, h = image.size
    size = max(w, h)
    new_image = Image.new('L', (size, size), 255)
    new_image.paste(image, ((size-w)//2, (size-h)//2))
    
    # Resize to 28x28
    try:
        image = new_image.resize((28, 28), Image.LANCZOS)
    except AttributeError:
        image = new_image.resize((28, 28), Image.ANTIALIAS)
    
    # Enhance contrast
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Ensure proper thresholding
    threshold = 127
    img_array = ((img_array > threshold) * 255).astype(np.uint8)
    
    # Invert if needed (ensure digits are white on black background like MNIST)
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Convert back to PIL Image
    image = Image.fromarray(img_array)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Add visualization of preprocessed image
    image.save('temp_preprocessed.png')
    
    return transform(image).unsqueeze(0)

def load_latest_model(model):
    # Look for all .pth files in the models directory
    model_files = glob.glob('models/*.pth')
    if not model_files:
        raise Exception("No trained model found!")
    
    # Find the most recent model file based on creation time
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")
    
    # Load the model weights
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    return model

def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.nn.functional.softmax(output, dim=1)
        prediction = output.argmax(1).item()
        confidence = probability[0][prediction].item() * 100
    return prediction, confidence

def display_prediction(image_path, prediction, confidence):
    # Display image with prediction
    image = Image.open(image_path).convert('L')
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Prediction: {prediction}\nConfidence: {confidence:.2f}%')
    plt.axis('off')
    
    # Display preprocessed image
    processed = preprocess_image(image_path)
    plt.subplot(1, 2, 2)
    plt.imshow(processed.squeeze(), cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    plt.show()

def test_multiple_images(image_folder):
    # Set up device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model architecture
    model = MNISTModel().to(device)
    
    # Load the trained weights into the model
    model = load_latest_model(model)
    model.eval()  # Set model to evaluation mode
    
    # Get all image files
    image_files = glob.glob(f'{image_folder}/*.png') + \
                 glob.glob(f'{image_folder}/*.jpg') + \
                 glob.glob(f'{image_folder}/*.jpeg')
    
    for image_path in image_files:
        try:
            # Process image
            image_tensor = preprocess_image(image_path).to(device)
            
            # Get prediction
            prediction, confidence = predict_image(model, image_tensor)
            
            # Display results
            print(f"\nImage: {os.path.basename(image_path)}")
            print(f"Predicted digit: {prediction}")
            print(f"Confidence: {confidence:.2f}%")
            
            display_prediction(image_path, prediction, confidence)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    # Create a test folder path
    test_folder = "test_images"
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"Created folder: {test_folder}")
        print("Please add your test images to this folder and run the script again")
    else:
        # Start testing images
        test_multiple_images(test_folder) 
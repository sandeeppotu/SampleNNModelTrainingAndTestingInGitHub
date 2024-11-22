import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from model import MNISTModel
import glob
import os
import numpy as np

def apply_augmentation(image):
    """Apply the same augmentations used in training"""
    aug_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(15, fill=0),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
            fill=0
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),
        transforms.ElasticTransform(alpha=30.0, sigma=3.0, fill=0),
        transforms.ToPILImage()
    ])
    return aug_transform(image)

def preprocess_image(image_path):
    # Open image and convert to grayscale
    image = Image.open(image_path).convert('L')
    raw_image = image.copy()
    
    # Add padding to make it square
    w, h = image.size
    size = max(w, h)
    new_image = Image.new('L', (size, size), 255)
    new_image.paste(image, ((size-w)//2, (size-h)//2))
    
    # Create augmented version
    augmented_image = apply_augmentation(new_image)
    
    # Resize to 28x28
    try:
        image = new_image.resize((28, 28), Image.LANCZOS)
    except AttributeError:
        image = new_image.resize((28, 28), Image.ANTIALIAS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Ensure proper thresholding
    threshold = 127
    img_array = ((img_array > threshold) * 255).astype(np.uint8)
    
    # Invert if needed
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Save preprocessed image
    preprocessed_image = Image.fromarray(img_array)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(preprocessed_image).unsqueeze(0), raw_image, augmented_image, preprocessed_image

def display_prediction(image_path, prediction, confidence):
    # Get all image versions
    tensor_image, raw_image, augmented_image, preprocessed_image = preprocess_image(image_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle(f'Prediction: {prediction} (Confidence: {confidence:.2f}%)', fontsize=14)
    
    # Plot raw image
    axes[0].imshow(raw_image, cmap='gray')
    axes[0].set_title('Raw Image')
    axes[0].axis('off')
    
    # Plot augmented image
    axes[1].imshow(augmented_image, cmap='gray')
    axes[1].set_title('Augmented')
    axes[1].axis('off')
    
    # Plot preprocessed image
    axes[2].imshow(preprocessed_image, cmap='gray')
    axes[2].set_title('Preprocessed')
    axes[2].axis('off')
    
    # Plot tensor visualization
    axes[3].imshow(tensor_image.squeeze().numpy(), cmap='gray')
    axes[3].set_title('Model Input')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    save_dir = 'prediction_results'
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f'{base_name}_prediction.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.nn.functional.softmax(output, dim=1)
        prediction = output.argmax(1).item()
        confidence = probability[0][prediction].item() * 100
    return prediction, confidence

def test_multiple_images(image_folder):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MNISTModel().to(device)
    model = load_latest_model(model)
    model.eval()
    
    # Create results directory
    results_dir = 'prediction_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all image files
    image_files = glob.glob(f'{image_folder}/*.png') + \
                 glob.glob(f'{image_folder}/*.jpg') + \
                 glob.glob(f'{image_folder}/*.jpeg')
    
    print("\nProcessing images...")
    for image_path in image_files:
        try:
            # Process image and get tensor
            image_tensor, _, _, _ = preprocess_image(image_path)
            image_tensor = image_tensor.to(device)
            
            # Get prediction
            prediction, confidence = predict_image(model, image_tensor)
            
            # Display results
            print(f"\nImage: {os.path.basename(image_path)}")
            print(f"Predicted digit: {prediction}")
            print(f"Confidence: {confidence:.2f}%")
            
            # Display and save visualization
            display_prediction(image_path, prediction, confidence)
            print(f"Prediction visualization saved in: prediction_results/")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

def load_latest_model(model):
    model_files = glob.glob('models/*.pth')
    if not model_files:
        raise Exception("No trained model found!")
    
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    return model

if __name__ == "__main__":
    # Create a test folder path
    test_folder = "test_images"
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"Created folder: {test_folder}")
        print("Please add your test images to this folder and run the script again")
    else:
        test_multiple_images(test_folder) 
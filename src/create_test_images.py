import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

def create_digit_image(digit, size=(128, 128), font_size=80):
    # Create white image with some padding
    image = Image.new('L', size, 'white')
    draw = ImageDraw.Draw(image)
    
    # Try different fonts
    try:
        # Try different font paths based on OS
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf"
        ]
        
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue
                
        if font is None:
            font = ImageFont.load_default()
            
    except:
        font = ImageFont.load_default()
    
    # Get text size
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate center position
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw the digit in black
    draw.text((x, y), text, fill='black', font=font)
    
    # Apply slight smoothing
    image = image.filter(ImageFilter.SMOOTH)
    
    # Ensure good contrast
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    return image

def main():
    # Create test_images directory if it doesn't exist
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
    
    print("Creating test images...")
    # Create sample images for digits 0-9
    for digit in range(10):
        image = create_digit_image(digit)
        # Save with high quality
        save_path = f'test_images/sample_digit_{digit}.png'
        image.save(save_path, quality=95)
        print(f"Created test image for digit {digit}: {save_path}")

if __name__ == "__main__":
    main() 
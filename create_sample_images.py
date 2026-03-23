#!/usr/bin/env python3
"""
Create sample images for testing the skin disease classification model
"""

import os
import numpy as np
from PIL import Image
import cv2

def create_sample_images():
    """Create sample skin disease images for testing"""
    
    # Create sample images for each disease
    diseases = ['acne', 'psoriasis', 'eczema']
    
    for disease in diseases:
        # Create 5 sample images per disease
        for i in range(5):
            # Create a synthetic skin disease image
            image = np.random.randint(180, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add disease-specific patterns
            if disease == 'acne':
                # Add red dots for acne
                for _ in range(15):
                    x, y = np.random.randint(20, 204, 2)
                    cv2.circle(image, (x, y), 3, (200, 50, 50), -1)
            
            elif disease == 'psoriasis':
                # Add patches for psoriasis
                for _ in range(3):
                    x, y = np.random.randint(20, 180, 2)
                    cv2.rectangle(image, (x, y), (x+30, y+30), (255, 200, 150), -1)
            
            elif disease == 'eczema':
                # Add irregular patches for eczema
                for _ in range(5):
                    x, y = np.random.randint(20, 180, 2)
                    cv2.ellipse(image, (x+15, y+15), (20, 15), 0, 0, 360, (255, 180, 180), -1)
            
            # Save image
            image_path = f"data/train/{disease}/sample_{i+1}.jpg"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # Convert to PIL and save
            pil_image = Image.fromarray(image)
            pil_image.save(image_path)
            print(f"Created {image_path}")
    
    # Create validation and test images
    for split in ['val', 'test']:
        for disease in diseases:
            for i in range(2):
                image = np.random.randint(180, 255, (224, 224, 3), dtype=np.uint8)
                
                # Add subtle patterns
                if disease == 'acne':
                    for _ in range(8):
                        x, y = np.random.randint(20, 204, 2)
                        cv2.circle(image, (x, y), 2, (200, 50, 50), -1)
                
                elif disease == 'psoriasis':
                    for _ in range(2):
                        x, y = np.random.randint(20, 180, 2)
                        cv2.rectangle(image, (x, y), (x+25, y+25), (255, 200, 150), -1)
                
                elif disease == 'eczema':
                    for _ in range(3):
                        x, y = np.random.randint(20, 180, 2)
                        cv2.ellipse(image, (x+12, y+12), (15, 10), 0, 0, 360, (255, 180, 180), -1)
                
                image_path = f"data/{split}/{disease}/sample_{i+1}.jpg"
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                
                pil_image = Image.fromarray(image)
                pil_image.save(image_path)
                print(f"Created {image_path}")
    
    print("\nSample images created successfully!")
    print("Dataset structure:")
    print("- Train: 15 images per class (45 total)")
    print("- Val: 2 images per class (6 total)")
    print("- Test: 2 images per class (6 total)")
    print("- Total: 57 images")

if __name__ == "__main__":
    create_sample_images()

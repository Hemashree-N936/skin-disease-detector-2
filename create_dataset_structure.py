#!/usr/bin/env python3
"""
Create dataset directory structure for skin disease classification
"""

import os

def create_dataset_structure():
    """Create the required directory structure"""
    base_dir = "data"
    
    # Define directory structure
    directories = [
        os.path.join(base_dir, "train", "acne"),
        os.path.join(base_dir, "train", "psoriasis"),
        os.path.join(base_dir, "train", "eczema"),
        os.path.join(base_dir, "val", "acne"),
        os.path.join(base_dir, "val", "psoriasis"),
        os.path.join(base_dir, "val", "eczema"),
        os.path.join(base_dir, "test", "acne"),
        os.path.join(base_dir, "test", "psoriasis"),
        os.path.join(base_dir, "test", "eczema"),
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("\nDataset structure created successfully!")
    print("Please add your skin disease images to the appropriate directories:")
    print("- data/train/acne/")
    print("- data/train/psoriasis/")
    print("- data/train/eczema/")
    print("- data/val/acne/")
    print("- data/val/psoriasis/")
    print("- data/val/eczema/")
    print("- data/test/acne/")
    print("- data/test/psoriasis/")
    print("- data/test/eczema/")

if __name__ == "__main__":
    create_dataset_structure()

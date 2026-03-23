# Bias-Mitigated CNN-GAN-ViT Hybrid for Multi-Class Detection of Acne, Psoriasis, and Eczema on Indian Dark Skin Tones

## Dataset Preparation Script

This repository contains a comprehensive Python script for preparing a balanced dataset focused on Indian dark skin tones for the detection of acne, psoriasis, and eczema.

## Features

- **Multi-source dataset loading**: Combines DermaCon-IN and Kaggle skin disease datasets
- **Class filtering**: Focuses exclusively on Acne, Psoriasis, and Eczema
- **Image preprocessing**: Resizes to 224x224 and applies ImageNet normalization
- **Stratified splitting**: 70% train, 15% validation, 15% test with class balance preservation
- **Comprehensive logging**: Detailed statistics and class imbalance reporting
- **Visualization**: Automatic generation of class distribution plots

## Project Structure

```
d:/skin_diseases_3/
├── skin_disease_dataset_preparation.py  # Main dataset preparation script
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── data/                               # Input datasets
│   ├── dermacon_in/                   # DermaCon-IN dataset
│   ├── kaggle_acne/                   # Kaggle acne dataset
│   ├── kaggle_psoriasis/              # Kaggle psoriasis dataset
│   └── kaggle_eczema/                 # Kaggle eczema dataset
├── dataset/                           # Output processed dataset
│   ├── train/
│   │   ├── acne/
│   │   ├── psoriasis/
│   │   └── eczema/
│   ├── val/
│   │   ├── acne/
│   │   ├── psoriasis/
│   │   └── eczema/
│   └── test/
│       ├── acne/
│       ├── psoriasis/
│       └── eczema/
├── logs/                              # Execution logs
└── stats/                             # Statistics and visualizations
```

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset Structure

Organize your raw datasets in the following structure:

```
data/
├── dermacon_in/
│   ├── acne/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── psoriasis/
│   └── eczema/
├── kaggle_acne/
├── kaggle_psoriasis/
└── kaggle_eczema/
```

### 2. Run the Dataset Preparation Script

```bash
python skin_disease_dataset_preparation.py
```

### 3. Check Results

After execution, you will find:

- **Processed dataset** in `dataset/` folder with train/val/test splits
- **Statistics** in `stats/dataset_statistics.json`
- **Visualizations** in `stats/class_distribution.png`
- **Execution logs** in `logs/dataset_preparation.log`

## Configuration

The script can be customized by modifying the `Config` class in the main script:

```python
class Config:
    # Dataset paths - update these to match your data locations
    DERMACON_IN_PATH = "data/dermacon_in"
    KAGGLE_ACNE_PATH = "data/kaggle_acne"
    KAGGLE_PSORIASIS_PATH = "data/kaggle_psoriasis"
    KAGGLE_ECZEMA_PATH = "data/kaggle_eczema"
    
    # Target classes (only these will be kept)
    TARGET_CLASSES = ["acne", "psoriasis", "eczema"]
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)
    
    # Dataset split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
```

## Key Features

### Class Mapping

The script automatically maps various class name variations to the target classes:

- **Acne**: acne, acne_vulgaris, pimple, pimples
- **Psoriasis**: psoriasis, psoriatic, plaque_psoriasis
- **Eczema**: eczema, dermatitis, atopic_dermatitis, contact_dermatitis

### Image Validation

- Corrupted image detection and removal
- Minimum size filtering (32x32 pixels)
- Automatic RGB conversion
- Support for multiple image formats (JPG, PNG, BMP, TIFF)

### Class Imbalance Handling

The script provides:
- Class distribution statistics
- Class weight calculations for training
- Imbalance ratio reporting
- Visualization of class distribution

### Logging and Monitoring

Comprehensive logging includes:
- Total images per class
- Processing progress
- Error reporting for corrupted images
- Final statistics summary

## Output Statistics Example

```
DATASET STATISTICS
==================================================
Total images: 15000
Class distribution:
  acne: 6000 (40.00%)
  psoriasis: 4500 (30.00%)
  eczema: 4500 (30.00%)
Class imbalance ratio: 1.33
Class weights:
  acne: 0.8333
  psoriasis: 1.1111
  eczema: 1.1111
```

## Next Steps

After dataset preparation, you can proceed with:

1. **Model Development**: Implement the CNN-GAN-ViT hybrid architecture
2. **Training**: Use the prepared dataset with class weights for bias mitigation
3. **Evaluation**: Test model performance on the held-out test set
4. **Deployment**: Integrate the trained model into your application

## Troubleshooting

### Common Issues

1. **Dataset path not found**: Update the paths in the `Config` class
2. **No images loaded**: Check that your dataset folders contain the expected class subfolders
3. **Memory issues**: Process datasets in smaller batches if needed
4. **Permission errors**: Ensure write permissions for output directories

### Error Handling

The script includes comprehensive error handling:
- Corrupted image detection
- Path validation
- File format checking
- Graceful failure with detailed logging

## Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **PIL (Pillow)**: Image processing
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Class weight calculation
- **OpenCV**: Additional image processing support

## License

This project is for research and educational purposes. Please ensure you have appropriate licenses for any datasets used.

## Citation

If you use this dataset preparation pipeline in your research, please cite:

```
Bias-Mitigated CNN-GAN-ViT Hybrid for Multi-Class Detection of Acne, Psoriasis, and Eczema on Indian Dark Skin Tones
Dataset Preparation Pipeline
```

## Contact

For questions or issues related to this dataset preparation script, please refer to the logging output and error messages for detailed troubleshooting information.

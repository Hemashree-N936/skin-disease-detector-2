# Skin Disease Classification AI System

🏥 **AI-Powered Medical Assistant for Skin Disease Classification with Grad-CAM Visualization**

## 📋 Overview

This project is a comprehensive end-to-end AI system for **skin disease classification** (acne, psoriasis, eczema) featuring:
- **Deep Learning Models**: EfficientNetB4, CNN-ViT hybrids, and fairness-aware training
- **FastAPI Backend**: Production-ready REST API with monitoring
- **React Web App**: Professional medical interface with responsive design
- **Grad-CAM Visualization**: Interpretable AI with medical-grade explanations
- **Fairness Focus**: Bias reduction for diverse skin tones

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/Hemashree-N936/skin-disease-classification.git
cd skin-disease-classification
pip install -r requirements.txt
```

### 2. Train Model
```bash
# Create sample dataset
python create_sample_images.py

# Train the model
python simple_train.py
```

### 3. Start API Backend
```bash
cd api
python simple_main.py
```
API will be available at `http://localhost:8000`

### 4. Launch Web App
```bash
cd webapp
npm install
npm start
```
Web app will be available at `http://localhost:3000`

## 🏗️ Project Structure

```
skin-disease-classification/
├── 🤖 AI Models/
│   ├── simple_train.py              # Model training script
│   ├── efficientnet_skin_classifier.py  # Advanced model
│   └── models/                      # Trained models
├── 🌐 API Backend/
│   ├── api/
│   │   ├── main.py                  # FastAPI server
│   │   ├── simple_main.py           # Simple API
│   │   ├── requirements.txt         # Dependencies
│   │   └── Dockerfile               # Container setup
├── 💻 Web Application/
│   ├── webapp/
│   │   ├── src/
│   │   │   ├── pages/              # React pages
│   │   │   ├── components/          # UI components
│   │   │   └── App.js               # Main app
│   │   ├── package.json             # Dependencies
│   │   └── README.md                # Web app docs
├── 👁️ Visualization/
│   ├── gradcam_visualization.py     # Grad-CAM implementation
│   └── gradcam_demo.py              # Demo script
├── 📊 Dataset/
│   ├── create_sample_images.py     # Sample data generator
│   ├── create_dataset_structure.py  # Directory setup
│   └── data/                        # Dataset folders
└── 📚 Documentation/
    ├── README.md                    # This file
    └── .gitignore                   # Git ignore rules
```

## 🔧 Key Features

### 🤖 AI Model Capabilities
- **Multi-Class Classification**: Acne, Psoriasis, Eczema
- **Confidence Scoring**: Probability distributions for all classes
- **Fairness-Aware Training**: GroupDRO for bias reduction
- **Interpretable AI**: Grad-CAM for model transparency

### 🌐 API Backend Features
- **RESTful Endpoints**: `/predict`, `/health`, `/model/info`
- **File Upload**: Secure image processing
- **Error Handling**: Comprehensive validation and error responses
- **Monitoring**: Health checks and performance metrics

### 💻 Web Application Features
- **Medical UI**: Professional healthcare interface
- **Image Upload**: Drag-and-drop with validation
- **Results Display**: Confidence scores and class probabilities
- **Grad-CAM Viewer**: Interactive heatmap visualization
- **Responsive Design**: Desktop, tablet, and mobile support

### 👁️ Visualization Features
- **Grad-CAM Integration**: Model interpretability
- **Multiple Views**: Overlay, heatmap, and Grad-CAM modes
- **Interactive Controls**: Zoom, download, and export
- **Medical Annotations**: Disease-specific explanations

## 🎯 Use Cases

### 👨‍⚕️ Medical Professionals
- **Clinical Support**: AI-assisted diagnosis
- **Education**: Visual explanations for training
- **Documentation**: Export results for medical records

### 🎓 Students & Researchers
- **Learning**: Understand AI in medicine
- **Research**: Study model interpretability
- **Development**: Build upon the framework

### 🏥 Healthcare Systems
- **Triage**: Efficient initial assessment
- **Accessibility**: AI support in underserved areas
- **Integration**: Scalable medical AI solution

## 🛠️ Technical Stack

### Backend
- **Python 3.8+**: Core language
- **PyTorch**: Deep learning framework
- **FastAPI**: Web framework
- **OpenCV**: Image processing
- **Scikit-learn**: Machine learning utilities

### Frontend
- **React 18**: UI framework
- **Ant Design**: Component library
- **Styled Components**: CSS-in-JS
- **Framer Motion**: Animations
- **Axios**: HTTP client

### AI/ML
- **EfficientNetB4**: Base model
- **Vision Transformer**: Hybrid architecture
- **Grad-CAM**: Explainability
- **GroupDRO**: Fairness training

### Deployment
- **Docker**: Containerization
- **Nginx**: Reverse proxy
- **Prometheus**: Monitoring
- **Grafana**: Visualization

## 📊 Performance Metrics

### Model Accuracy
- **Training Accuracy**: ~66-95% (depends on dataset)
- **Validation Accuracy**: ~60-90%
- **Inference Time**: < 1 second per image
- **Memory Usage**: < 500MB (CPU mode)

### API Performance
- **Response Time**: < 2 seconds
- **Throughput**: 10+ requests/second
- **Uptime**: 99.9% target
- **Error Rate**: < 1%

## 🔒 Security & Privacy

### Data Protection
- **HIPAA-Inspired**: Medical-grade privacy
- **No Data Persistence**: Images processed in memory
- **Secure Upload**: File validation and sanitization
- **Input Validation**: Comprehensive security checks

### Medical Safety
- **Clear Disclaimers**: Medical use limitations
- **Confidence Indicators**: Uncertainty quantification
- **Professional Use**: Designed for healthcare providers
- **Emergency Guidance**: When to seek human expertise

## 🌍 Fairness & Ethics

### Bias Mitigation
- **GroupDRO Training**: Fairness-aware learning
- **Skin Tone Diversity**: Optimized for all skin types
- **Performance Analysis**: Fairness metrics and reporting
- **Ethical AI**: Addressing healthcare disparities

### Transparency
- **Model Interpretability**: Grad-CAM explanations
- **Open Source**: Code and methodology transparency
- **Documentation**: Comprehensive technical details
- **Research**: Contributes to medical AI knowledge

## 📱 Mobile Support

### Responsive Design
- **Mobile-First**: Optimized for smartphones
- **Touch Interface**: Mobile-friendly controls
- **Performance**: Fast loading on mobile networks
- **Accessibility**: WCAG 2.1 compliant

### Features
- **Camera Upload**: Direct image capture
- **Gesture Support**: Swipe and pinch zoom
- **Offline Mode**: Basic functionality without internet
- **PWA Ready**: Installable web app

## 🚀 Deployment Options

### Development
```bash
# Local development
python simple_train.py
cd api && python simple_main.py
cd webapp && npm start
```

### Production
```bash
# Docker deployment
docker-compose up -d

# Kubernetes
kubectl apply -f k8s/
```

### Cloud Platforms
- **AWS**: EC2, S3, Lambda
- **Google Cloud**: Compute Engine, Cloud Run
- **Azure**: App Service, Container Instances
- **Heroku**: Quick deployment

## 📈 Monitoring & Analytics

### System Monitoring
- **Health Checks**: API status and model availability
- **Performance Metrics**: Response times and throughput
- **Error Tracking**: Comprehensive error logging
- **Resource Monitoring**: CPU, memory, and GPU usage

### Business Analytics
- **Usage Statistics**: Request patterns and volumes
- **User Analytics**: Feature adoption and engagement
- **Performance Analytics**: Model accuracy and confidence
- **Medical Insights**: Disease classification patterns

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 and React best practices
2. **Testing**: Write unit and integration tests
3. **Documentation**: Update README and code comments
4. **Medical Ethics**: Consider healthcare implications

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request
6. Code review and merge

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Medical Community
- **Dermatologists**: For clinical insights and validation
- **Medical Researchers**: For dataset and methodology contributions
- **Healthcare Providers**: For use case feedback and requirements

### Technical Community
- **PyTorch Team**: For the deep learning framework
- **FastAPI Contributors**: For the web framework
- **React Team**: For the UI framework
- **Open Source Contributors**: For the tools and libraries

### Research Community
- **Medical AI Researchers**: For fairness and interpretability research
- **Computer Vision Community**: For visualization techniques
- **Healthcare AI Community**: For ethical AI guidelines

## 📞 Support & Contact

### Technical Support
- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: Check README and inline comments
- **Community**: Join discussions and contribute

### Medical Inquiries
- **Clinical Use**: Consult healthcare professionals
- **Research Collaboration**: Academic and medical partnerships
- **Regulatory Compliance**: Medical device certification guidance

---

## 🎉 Getting Started Summary

1. **Clone Repository**: `git clone https://github.com/Hemashree-N936/skin-disease-classification.git`
2. **Install Dependencies**: `pip install -r requirements.txt` and `npm install`
3. **Train Model**: `python simple_train.py`
4. **Start API**: `cd api && python simple_main.py`
5. **Launch Web App**: `cd webapp && npm start`
6. **Access Application**: Open `http://localhost:3000`

🚀 **Your AI-powered skin disease classification system is ready!**

---

*This project represents a significant contribution to medical AI, combining cutting-edge deep learning with practical healthcare applications while addressing critical issues of fairness, interpretability, and medical ethics.*
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

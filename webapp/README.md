# SkinAI - React Web Application

A modern, responsive React web application for AI-powered skin disease classification with Grad-CAM visualization.

## Features

- 🏥 **Medical AI Interface**: Clean, professional UI designed for medical professionals
- 📸 **Image Upload**: Drag-and-drop or click-to-upload skin disease images
- 🤖 **AI Classification**: Real-time disease prediction with confidence scores
- 👁️ **Grad-CAM Visualization**: Interactive heatmap overlays showing model focus areas
- 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- 🎨 **Modern UI**: Beautiful gradient design with smooth animations
- 🔒 **Privacy-First**: Secure image handling with medical compliance considerations

## Quick Start

### Prerequisites

- Node.js 16+ installed
- FastAPI backend running on `http://localhost:8000`

### Installation

```bash
cd webapp
npm install
```

### Development

```bash
npm start
```

The app will be available at `http://localhost:3000`

### Production Build

```bash
npm run build
```

## Application Structure

```
webapp/
├── public/
│   ├── index.html          # Main HTML file
│   └── favicon.ico         # App icon
├── src/
│   ├── components/         # Reusable components
│   │   ├── Header.js      # Navigation header
│   │   ├── Footer.js      # Footer component
│   │   ├── ImagePreview.js # Image upload preview
│   │   ├── PredictionResults.js # Prediction display
│   │   ├── GradCAMVisualization.js # Heatmap viewer
│   │   └── DiseaseInfo.js # Disease information
│   ├── pages/             # Page components
│   │   ├── Home.js        # Landing page
│   │   ├── Classifier.js  # Main classification interface
│   │   └── About.js       # About page
│   ├── App.js             # Main app component
│   ├── index.js           # App entry point
│   └── index.css          # Global styles
├── package.json           # Dependencies
└── README.md             # This file
```

## Components Overview

### Main Components

#### **Classifier** (`pages/Classifier.js`)
- Main application interface
- Image upload with drag-and-drop
- Real-time prediction results
- Grad-CAM visualization
- Disease information display

#### **ImagePreview** (`components/ImagePreview.js`)
- Image upload component
- File validation and preview
- Upload progress tracking
- Analysis options (quick vs. with Grad-CAM)

#### **PredictionResults** (`components/PredictionResults.js`)
- Displays prediction confidence
- Shows class probabilities
- Processing time information
- Visual confidence indicators

#### **GradCAMVisualization** (`components/GradCAMVisualization.js`)
- Interactive heatmap viewer
- Multiple visualization modes
- Zoom controls
- Download functionality

#### **DiseaseInfo** (`components/DiseaseInfo.js`)
- Disease-specific information
- Key features and symptoms
- Treatment overview
- Medical disclaimer

## UI Features

### Design System

- **Colors**: Medical-grade color palette
- **Typography**: Clean, readable fonts optimized for medical use
- **Spacing**: Consistent spacing system
- **Animations**: Smooth transitions and micro-interactions
- **Responsive**: Mobile-first responsive design

### User Experience

- **Intuitive Navigation**: Clear menu structure and breadcrumbs
- **Visual Feedback**: Loading states, progress indicators, success/error messages
- **Accessibility**: WCAG 2.1 compliant design
- **Performance**: Optimized for fast loading and smooth interactions

### Medical Considerations

- **Professional Design**: Clean, clinical interface suitable for medical professionals
- **Clear Information**: Easy-to-understand results and explanations
- **Privacy Focus**: Secure image handling and data protection
- **Medical Disclaimer**: Clear guidance on medical use limitations

## API Integration

### Backend Communication

The app integrates with the FastAPI backend through:

```javascript
// Quick prediction
POST /predict
Content-Type: multipart/form-data
Body: image file

// Prediction with Grad-CAM
POST /predict-with-gradcam
Content-Type: multipart/form-data
Body: image file
```

### Response Handling

```javascript
// Prediction response
{
  "class_label": "acne",
  "confidence": 0.847,
  "probabilities": {
    "acne": 0.847,
    "psoriasis": 0.092,
    "eczema": 0.061
  },
  "processing_time": 0.123,
  "timestamp": "2025-03-22T12:00:00.000Z"
}

// Grad-CAM response
{
  "gradcam_image": "base64_encoded_image",
  "heatmap_image": "base64_encoded_image",
  "overlay_image": "base64_encoded_image",
  "class_label": "acne",
  "confidence": 0.847,
  "class_description": "Inflammatory skin condition..."
}
```

## Responsive Design

### Breakpoints

- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

### Mobile Optimizations

- Touch-friendly interface
- Simplified navigation
- Optimized image handling
- Compact layouts
- Gesture support

## Performance Features

### Optimizations

- **Code Splitting**: Lazy loading of components
- **Image Optimization**: Efficient image handling and compression
- **Caching**: Browser caching for static assets
- **Bundle Optimization**: Minimized production builds

### Monitoring

- **Error Tracking**: Comprehensive error handling and logging
- **Performance Metrics**: Loading time and interaction tracking
- **User Analytics**: Usage patterns and feature adoption

## Security Features

### Data Protection

- **Secure Upload**: File type and size validation
- **Privacy Compliance**: HIPAA-inspired data handling
- **Input Validation**: Comprehensive input sanitization
- **CSRF Protection**: Cross-site request forgery prevention

### Medical Safety

- **Disclaimer**: Clear medical use limitations
- **Professional Use**: Designed for healthcare professionals
- **Accuracy Information**: Confidence scores and limitations
- **Emergency Guidance: When to seek human medical advice

## Development

### Local Development

```bash
# Install dependencies
npm install

# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build
```

### Environment Variables

```bash
# API configuration
REACT_APP_API_URL=http://localhost:8000

# Feature flags
REACT_APP_ENABLE_ANALYTICS=true
REACT_APP_ENABLE_DEBUG=false
```

### Code Quality

```bash
# Code formatting
npm run format

# Linting
npm run lint

# Type checking
npm run type-check
```

## Deployment

### Production Build

```bash
# Create optimized build
npm run build

# Test production build locally
serve -s build
```

### Deployment Options

1. **Static Hosting**: Netlify, Vercel, GitHub Pages
2. **CDN**: CloudFront, Cloudflare
3. **Server**: Nginx, Apache
4. **Container**: Docker, Kubernetes

### Environment Configuration

```bash
# Production
NODE_ENV=production
REACT_APP_API_URL=https://api.skindisease-ai.com

# Staging
NODE_ENV=staging
REACT_APP_API_URL=https://staging-api.skindisease-ai.com
```

## Browser Support

- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+
- **Mobile Safari**: iOS 14+
- **Chrome Mobile**: Android 10+

## Accessibility

### WCAG 2.1 Compliance

- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Readers**: ARIA labels and semantic HTML
- **Color Contrast**: AA level contrast ratios
- **Focus Management**: Clear focus indicators
- **Text Scaling**: Support for 200% zoom

### Medical Accessibility

- **High Contrast**: Enhanced contrast for medical professionals
- **Large Text**: Readable font sizes for clinical use
- **Clear Instructions**: Step-by-step guidance
- **Error Prevention**: Clear error messages and recovery

## Testing

### Test Suite

```bash
# Unit tests
npm test

# Integration tests
npm run test:integration

# E2E tests
npm run test:e2e

# Accessibility tests
npm run test:a11y
```

### Test Coverage

- **Components**: 90%+ coverage
- **User Flows**: Critical path testing
- **API Integration**: Backend communication
- **Responsive Design**: Multiple device testing

## Troubleshooting

### Common Issues

**API Connection Error:**
```bash
# Check backend is running
curl http://localhost:8000/health

# Update API URL
REACT_APP_API_URL=http://localhost:8000
```

**Image Upload Issues:**
- Check file size (< 10MB)
- Verify file format (JPG, PNG, BMP, TIFF)
- Ensure proper permissions

**Performance Issues:**
- Clear browser cache
- Check network connection
- Verify API response times

### Debug Mode

```bash
# Enable debug logging
REACT_APP_DEBUG=true npm start

# View console logs
# Check network tab in browser dev tools
```

## Contributing

### Development Guidelines

1. **Code Style**: Follow established patterns
2. **Components**: Keep components small and focused
3. **Testing**: Write tests for new features
4. **Documentation**: Update README and comments
5. **Accessibility**: Ensure WCAG compliance

### Pull Request Process

1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit pull request
5. Code review and merge

## License

This project is part of the skin disease classification research initiative.

---

## Support

For technical support or questions:

1. Check the troubleshooting section
2. Review the API documentation
3. Consult the component documentation
4. Contact the development team

## Future Enhancements

### Planned Features

- **Multi-language Support**: Internationalization
- **Offline Mode**: PWA capabilities
- **Advanced Analytics**: Usage insights
- **Integration**: EHR system connectivity
- **Mobile App**: Native mobile applications

### Research Integration

- **Model Updates**: Latest AI model versions
- **Clinical Validation**: Medical research integration
- **User Studies**: Healthcare professional feedback
- **Regulatory Compliance**: Medical device certification

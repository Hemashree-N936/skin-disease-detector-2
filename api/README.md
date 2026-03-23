# Skin Disease Classification API

Production-ready FastAPI backend for skin disease classification with Grad-CAM visualization.

## Features

- 🏥 **Medical AI**: Trained on acne, psoriasis, and eczema classification
- 🔍 **Grad-CAM Visualization**: Model interpretability with heatmaps
- 🚀 **High Performance**: Optimized for production deployment
- 📊 **Monitoring**: Prometheus metrics and Grafana dashboards
- 🛡️ **Security**: Rate limiting, CORS, and input validation
- 📚 **Documentation**: Auto-generated API docs (Swagger/OpenAPI)
- 🐳 **Docker Ready**: Containerized deployment with Docker Compose

## Quick Start

### 1. Install Dependencies

```bash
cd api
pip install -r requirements.txt
```

### 2. Start Development Server

```bash
python main.py --reload
```

The API will be available at `http://localhost:8000`

### 3. Test the API

```bash
python test_api.py
```

## API Endpoints

### Health Check
```http
GET /health
```

Returns API health status and model information.

### Model Information
```http
GET /model/info
```

Returns detailed model configuration and class information.

### Disease Prediction
```http
POST /predict
```

Upload an image and get disease prediction.

**Request:**
- `file`: Image file (JPG, PNG, etc.) - Max 10MB

**Response:**
```json
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
```

### Prediction with Grad-CAM
```http
POST /predict-with-gradcam
```

Upload an image and get prediction with Grad-CAM visualization.

**Response:**
```json
{
  "gradcam_image": "base64_encoded_image",
  "heatmap_image": "base64_encoded_image", 
  "overlay_image": "base64_encoded_image",
  "class_label": "acne",
  "confidence": 0.847,
  "class_description": "Inflammatory skin condition with comedones..."
}
```

### Metrics
```http
GET /metrics
```

Prometheus metrics endpoint for monitoring.

## Model Loading

The API automatically loads the best available model in this priority order:

1. **GroupDRO Model** (`models_groupdro/best_groupdro_model.pth`) - Fairness-aware
2. **Hybrid Model** (`models_hybrid/best_hybrid_model.pth`) - CNN-ViT
3. **Segmented Model** (`models_segmented/best_model.pth`) - SAM-enhanced
4. **Augmented Model** (`models_augmented/best_model.pth`) - GAN-augmented
5. **Standard Model** (`models/best_model.pth`) - Baseline

## Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

Services included:
- **API**: Main application (port 8000)
- **Nginx**: Reverse proxy (port 80/443)
- **Redis**: Caching (port 6379)
- **Prometheus**: Monitoring (port 9090)
- **Grafana**: Visualization (port 3000)

### Production Server

```bash
# Production mode with multiple workers
python main.py --workers 4

# Or with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics`:
- `predictions_total`: Total predictions by class and status
- `prediction_duration_seconds`: Prediction processing time
- `uploads_total`: Upload attempts by status

### Grafana Dashboard

Access Grafana at `http://localhost:3000` (admin/admin)

Pre-configured dashboards:
- API Performance
- Prediction Statistics
- Error Rates
- System Health

## Security Features

- **Rate Limiting**: 10 requests/second per IP
- **File Size Limit**: 10MB maximum upload size
- **Input Validation**: Image format and size validation
- **CORS**: Configurable cross-origin resource sharing
- **Security Headers**: X-Frame-Options, CSP, XSS protection
- **HTTPS**: SSL/TLS support with Nginx

## API Documentation

### Swagger UI
Interactive API documentation at `http://localhost:8000/docs`

### ReDoc
Alternative documentation at `http://localhost:8000/redoc`

## Configuration

### Environment Variables

```bash
# API Configuration
ENVIRONMENT=production
LOG_LEVEL=info
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Model Configuration
MODEL_PATH=models/best_model.pth
DEVICE=auto

# Redis Configuration
REDIS_URL=redis://localhost:6379
```

### Nginx Configuration

The Nginx reverse proxy provides:
- Load balancing
- SSL termination
- Rate limiting
- Static file serving
- Security headers

## Performance Optimization

### Caching
- Redis for session storage
- Static file caching with Nginx
- Model preloading and GPU optimization

### Scalability
- Multi-process deployment
- Load balancing
- Horizontal scaling support
- Resource monitoring

## Testing

### Run Tests

```bash
# Run all tests
python test_api.py

# Run specific tests
python test_api.py --test health
python test_api.py --test predict
python test_api.py --test gradcam
python test_api.py --test errors
python test_api.py --test performance
```

### Test Coverage

The test suite covers:
- ✅ Health check endpoint
- ✅ Model information endpoint
- ✅ Disease prediction
- ✅ Grad-CAM visualization
- ✅ Error handling
- ✅ Performance benchmarks

## Troubleshooting

### Common Issues

**Model not found:**
```bash
# Ensure models are in the correct location
ls -la models/
ls -la models_groupdro/
ls -la models_hybrid/
```

**CUDA out of memory:**
```bash
# Use CPU mode
python main.py --device cpu
```

**Port already in use:**
```bash
# Use different port
python main.py --port 8001
```

### Logs

Check logs for debugging:
```bash
# Application logs
tail -f logs/app.log

# Nginx logs
docker-compose logs nginx

# API logs
docker-compose logs api
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone <repository>
cd skin_diseases_3/api

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio httpx black flake8

# Run development server
python main.py --reload
```

### Code Quality

```bash
# Code formatting
black main.py

# Linting
flake8 main.py

# Type checking
mypy main.py
```

### API Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=api tests/
```

## Production Checklist

Before deploying to production:

- [ ] Train and save model in appropriate directory
- [ ] Test all API endpoints
- [ ] Configure SSL certificates
- [ ] Set up monitoring and alerting
- [ ] Configure rate limiting
- [ ] Set up log rotation
- [ ] Test error handling
- [ ] Performance testing
- [ ] Security audit

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Check the application logs
4. Run the test suite to verify functionality

## License

This project is part of the skin disease classification research initiative.

---

## API Usage Examples

### Python Client

```python
import requests

# Upload image for prediction
with open('skin_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted: {result['class_label']}")
        print(f"Confidence: {result['confidence']:.3f}")
```

### JavaScript Client

```javascript
// Upload image using FormData
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Predicted:', data.class_label);
    console.log('Confidence:', data.confidence);
});
```

### cURL Client

```bash
# Upload image for prediction
curl -X POST \
  -F "file=@skin_image.jpg" \
  http://localhost:8000/predict
```

## Version History

- **v1.0.0**: Initial release with FastAPI backend
- **v1.1.0**: Added Grad-CAM visualization
- **v1.2.0**: Production optimization and monitoring
- **v1.3.0**: Docker deployment and security enhancements

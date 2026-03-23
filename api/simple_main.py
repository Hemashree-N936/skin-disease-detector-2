#!/usr/bin/env python3
"""
Simple FastAPI backend for skin disease classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from PIL import Image
import io
import torchvision.transforms as transforms
import uvicorn
import os
import sys

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_train import SimpleSkinClassifier

app = FastAPI(title="Skin Disease Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and setup transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleSkinClassifier(num_classes=3)

# Load trained model if available
model_path = "../models/simple_best_model.pth"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {model_path}")
    print(f"Best validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
else:
    print(f"Warning: No trained model found at {model_path}")
    print("Please train the model first using: python simple_train.py")

model.to(device)
model.eval()

# Setup transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['acne', 'psoriasis', 'eczema']

@app.get("/")
async def root():
    return {
        "message": "Skin Disease Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(model_path),
        "device": str(device),
        "classes": class_names
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict skin disease from uploaded image
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        # Create response
        class_probabilities = {
            class_names[i]: float(probabilities[0, i].item())
            for i in range(len(class_names))
        }
        
        return {
            "class_label": class_names[predicted_class_idx],
            "confidence": confidence,
            "probabilities": class_probabilities,
            "filename": file.filename,
            "model_type": "Simple CNN",
            "device": str(device)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

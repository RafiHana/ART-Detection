from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import logging

from model_loader import ModelLoader
from image_utils import process_image, validate_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Art Detection API",
    description="AI-Powered Painting Detection System using EfficientNet-B0 and FFT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

model_loader = ModelLoader()

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Loading model")
        model_loader.load_model()
        logger.info("Model loaded success")
    except Exception as e:
        logger.error(f"Error model: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = static_path / "index.html"
    return HTMLResponse(content=html_file.read_text(), status_code=200)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded,
        "message": "Art Detection API is running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, or JPG)"
            )
        
        image_bytes = await file.read()
        
        is_valid, error_msg = validate_image(image_bytes)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"Processing image: {file.filename}")
        processed_image = process_image(image_bytes)
        
        prediction, confidence, probabilities = model_loader.predict(processed_image)
        
        response = {
            "prediction": prediction, 
            "confidence": float(confidence),
            "probabilities": {
                "real": float(probabilities[0]),
                "ai": float(probabilities[1])
            },
            "filename": file.filename
        }
        
        logger.info(f"Prediction: {prediction} with confidence: {confidence:.4f}")
        return JSONResponse(content=response)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "EfficientNet-B0 with FFT",
        "input_shape": "224x224x3",
        "classes": ["Real Painting", "AI-Generated"],
        "framework": "TensorFlow/Keras",
        "accuracy": "95.8%",
        "dataset_size": "22,000+ images"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,  
        reload=True
    )
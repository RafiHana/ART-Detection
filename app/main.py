from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from model_loader import ModelLoader
from image_utils import process_image, validate_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_loader = ModelLoader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading model")
        model_loader.load_model()
        logger.info("Model loaded success")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    pass

app = FastAPI(
    title="Art Detection API",
    description="EfficientNet-B0 + FFT (PyTorch Version)",
    version="1.0.0",
    lifespan=lifespan 
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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = static_path / "index.html"
    return HTMLResponse(content=html_file.read_text(), status_code=200)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
        is_valid, error_msg = validate_image(image_bytes)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Proses (Resize -> FFT -> Tensor)
        processed_tensor = process_image(image_bytes)
        
        # Prediksi
        prediction, confidence, probs = model_loader.predict(processed_tensor)
        
        return JSONResponse({
            "prediction": prediction, 
            "confidence": float(confidence),
            "probabilities": {
                "real": float(probs['real']),
                "ai": float(probs['ai'])
            },
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
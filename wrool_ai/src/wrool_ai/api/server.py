from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from ..utils.config_loader import ConfigLoader
from ..modules.nlp.text_classifier import TextClassifier

app = FastAPI(title="Wrool-AI API", version="1.0.0")

# Load configuration
config_loader = ConfigLoader()
config = config_loader.load_config()

# Initialize model
model = TextClassifier(config["model"])

class TextRequest(BaseModel):
    text: str
    model_name: Optional[str] = None

class BatchRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: List[float]

@app.get("/")
async def root():
    return {"message": "Wrool-AI API Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    try:
        result = model.predict(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(request: BatchRequest):
    try:
        results = model.predict_batch(request.texts)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_server():
    """Run the FastAPI server"""
    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        workers=config["api"]["workers"]
    )

if __name__ == "__main__":
    run_server()
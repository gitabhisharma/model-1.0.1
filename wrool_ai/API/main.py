from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import json

app = FastAPI(title="AI API Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    class_id: int


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    processed_text: str
    sentiment: str
    confidence: float


# Simple AI model (example)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Load or initialize your model
model = SimpleCNN(num_classes=10)
# model.load_state_dict(torch.load('model.pth'))
model.eval()


@app.get("/")
async def root():
    return {"message": "AI API Server is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "cuda_available": torch.cuda.is_available()}


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((32, 32))

        # Convert to tensor
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

        return PredictionResponse(
            prediction=classes[predicted.item()],
            confidence=confidence.item(),
            class_id=predicted.item()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/text", response_model=TextResponse)
async def analyze_text(request: TextRequest):
    try:
        # Simple sentiment analysis (replace with your NLP model)
        text = request.text.lower()
        positive_words = ['good', 'great', 'excellent', 'awesome', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'sad', 'angry']

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        if positive_count > negative_count:
            sentiment = "positive"
            confidence = positive_count / (positive_count + negative_count + 1e-5)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = negative_count / (positive_count + negative_count + 1e-5)
        else:
            sentiment = "neutral"
            confidence = 0.5

        return TextResponse(
            processed_text=text,
            sentiment=sentiment,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
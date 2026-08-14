"""
FastAPI application for MNIST digit prediction using the trained PyTorch CNN.

Expected model file:
    mnist_cnn_pytorch_final.pth

Run:
    pip install fastapi uvicorn torch torchvision pillow python-multipart
    uvicorn app:app --reload

Swagger UI:
    http://127.0.0.1:8000/docs
"""

from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms


# ============================================================
# 1. CNN architecture
#    This MUST match the architecture used during training.
# ============================================================

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 2. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mnist_cnn_pytorch_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 3. Load trained model
# ============================================================

model = ImprovedCNN().to(device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Place mnist_cnn_pytorch_final.pth in the same directory as app.py."
    )

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
)

model.load_state_dict(state_dict)
model.eval()


# ============================================================
# 4. Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


# ============================================================
# 5. FastAPI application
# ============================================================

app = FastAPI(
    title="MNIST CNN Prediction API",
    description="FastAPI inference service for a PyTorch MNIST CNN.",
    version="1.0.0",
)


# ============================================================
# 6. CORS configuration
# ============================================================
#
# For development, common frontend ports are included.
#
# Change this list when deploying your application.
#
# For example:
# allow_origins = [
#     "https://your-frontend-domain.com"
# ]
#
# Do NOT use allow_origins=["*"] together with credentials=True.
# ============================================================

allow_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# 7. Health check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "MNIST ImprovedCNN",
        "device": str(device),
    }


# ============================================================
# 8. Prediction endpoint
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and predict the handwritten digit.

    Returns:
        predicted_digit
        confidence
        probabilities
    """

    # Basic file-type validation
    allowed_types = {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                "Please upload PNG, JPEG, or WebP."
            ),
        )

    try:
        image_bytes = await file.read()

        image = Image.open(BytesIO(image_bytes)).convert("L")

        # MNIST images normally have a white background and dark digit.
        # If a user uploads a typical black-background/white-digit image,
        # uncomment the next line if required:
        #
        # image = ImageOps.invert(image)

        image_tensor = transform(image)

        # Add batch dimension:
        # [1, 28, 28] -> [1, 1, 28, 28]
        image_tensor = image_tensor.unsqueeze(0).to(device)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded image: {exc}",
        ) from exc

    # ========================================================
    # Inference
    # ========================================================

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        predicted_digit = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        confidence = probabilities[
            0, predicted_digit
        ].item()

    probability_dict = {
        str(i): round(probabilities[0, i].item(), 6)
        for i in range(10)
    }

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 6),
        "confidence_percent": round(confidence * 100, 2),
        "probabilities": probability_dict,
    }


# ============================================================
# 9. Root endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "MNIST CNN Prediction API",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
    }

import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

class ResNet18ProjectionEncoder(nn.Module):
    def __init__(self, projection_dim=128, pretrained=False):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        projected = self.projection(features)
        return F.normalize(projected, p=2, dim=1)

# Device config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 224

# Preprocessing Transform
preprocess_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Model Loader
MODEL_WEIGHTS_PATH = "/home/tndor/Documents/Work/Classwork/semester-4/main-challenge/data/outputs/multidataset_standardized_resnet18/models/best_multidataset_standardized_resnet18.pt"

def load_retrieval_model():
    model = ResNet18ProjectionEncoder(projection_dim=128, pretrained=False)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Loading weights from {MODEL_WEIGHTS_PATH}")
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"WARNING: Weights file not found at {MODEL_WEIGHTS_PATH}. Running with randomized weights.")
    model.to(DEVICE)
    model.eval()
    return model

# Instantiate model globally
retrieval_model = load_retrieval_model()

app = FastAPI(title="Sketch2Face Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


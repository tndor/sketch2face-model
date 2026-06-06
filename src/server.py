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

# Global gallery cache
# Format: { folder_path: { "ids": [...], "embeddings": Tensor of shape (N, 128), "paths": [...] } }
GALLERY_CACHE = {}

def scan_and_cache_gallery(folder_path_str: str) -> dict:
    path = Path(folder_path_str)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Directory {folder_path_str} does not exist.")
    
    # Check cache
    if folder_path_str in GALLERY_CACHE:
        return GALLERY_CACHE[folder_path_str]
        
    image_extensions = {".png", ".jpg", ".jpeg"}
    img_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                img_paths.append(Path(root) / file)
                
    img_paths.sort()
    
    if not img_paths:
        raise ValueError(f"No valid images found in directory {folder_path_str}")
        
    ids = []
    embeddings_list = []
    
    with torch.no_grad():
        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = preprocess_transform(img).unsqueeze(0).to(DEVICE)
                embedding = retrieval_model(tensor)
                
                # Extract ID (e.g. filename without extension)
                subject_id = img_path.stem
                
                ids.append(subject_id)
                embeddings_list.append(embedding.cpu())
            except Exception as e:
                print(f"Skipping corrupt image {img_path}: {e}")
                
    if not embeddings_list:
        raise ValueError(f"Could not encode any images in {folder_path_str}")
        
    embeddings = torch.cat(embeddings_list, dim=0)
    
    data = {
        "ids": ids,
        "embeddings": embeddings,
        "paths": [str(p) for p in img_paths]
    }
    GALLERY_CACHE[folder_path_str] = data
    print(f"Cached {len(ids)} gallery images from {folder_path_str}")
    return data

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


import os
import io
import base64
import typing
import torch
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from PIL import Image
from torchvision import transforms

# --- MOCK IMPORT VS REAL IMPORT ---
# Swap this comment out when deploying with the real model
from .model import SymFormer as ModelClass
# from .model import MockSymFormer as ModelClass

# Constants & Class Labels
MODEL_PATH = os.getenv("MODEL_PATH", "backend/best_model.pth")
CLASS_LABELS = [
    {"id": 0, "name": "Healthy", "color": "#22c55e", "desc": "No abnormalities detected."},
    {"id": 1, "name": "Sick (Non-TB)", "color": "#f59e0b", "desc": "Abnormalities consistent with non-TB illness found."},
    {"id": 2, "name": "Active TB", "color": "#ef4444", "desc": "Active tuberculosis detected."},
    {"id": 3, "name": "Latent TB", "color": "#f97316", "desc": "Markers for latent tuberculosis observed."}
]

# Image processing configuration (adjust ImageNet norms if required differently)
IMG_SIZE = 512
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize App
app = FastAPI(title="SymFormer Medical AI API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global states
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    score: float

class PredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    label_id: int
    label_name: str
    label_color: str
    label_desc: str
    confidence: float
    probabilities: typing.List[float]
    has_lesion: bool
    heatmap_b64: str
    bboxes: typing.List[BBox]

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = ModelClass(num_classes_test=4)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Loaded real weights from {MODEL_PATH}")
        else:
            print(f"File {MODEL_PATH} not found. Using randomly initialized (Mock) weights.")
        model.to(device)
        model.eval()
        print(f"Model successfully loaded to {device}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "device": str(device)}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # Read the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    orig_w, orig_h = image.size
    
    # Run through transform
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, det_map = model(input_tensor)
        
    logits = logits.squeeze(0).cpu()  # (4,)
    det_map = det_map.squeeze(0).cpu() # (H, W) or (1, H, W)
    if det_map.dim() == 3 and det_map.shape[0] == 1:
        det_map = det_map.squeeze(0)
        
    # Calculate probabilities
    probs = torch.softmax(logits, dim=0).numpy()
    pred_idx = int(torch.argmax(logits, dim=0).item())
    
    # Process Heatmap
    heatmap_sigmoid = torch.sigmoid(det_map).numpy()
    
    # Resize to original image dims
    heatmap_resized = cv2.resize(heatmap_sigmoid, (orig_w, orig_h))
    
    # Bounding boxes threshold logic (activation > 0.35)
    heatmap_thresh = (heatmap_resized > 0.35).astype(np.uint8) * 255
    contours, _ = cv2.findContours(heatmap_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    has_lesion = False
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        region = heatmap_resized[y:y+h, x:x+w]
        score = float(np.max(region)) if region.size > 0 else 0.0
        
        bboxes.append(BBox(x=x, y=y, w=w, h=h, score=score))
        if score > 0.35:
            has_lesion = True
            
    # Visualize heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_INFERNO)
    
    # Original image array for alpha blending
    orig_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Alpha blend: 55% original, 45% heatmap
    blended = cv2.addWeighted(orig_array, 0.55, heatmap_color, 0.45, 0)
    
    # Draw red bboxes
    for bbox in bboxes:
        cv2.rectangle(blended, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), (0, 0, 255), 2)
        
    # Encode memory to Base64
    _, buffer = cv2.imencode('.png', blended)
    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Populate the response
    selected_label = CLASS_LABELS[pred_idx]
    
    return PredictResponse(
        label_id=selected_label["id"],
        label_name=selected_label["name"],
        label_color=selected_label["color"],
        label_desc=selected_label["desc"],
        confidence=round(float(probs[pred_idx] * 100), 1),
        probabilities=[round(float(p * 100), 1) for p in probs],
        has_lesion=has_lesion,
        heatmap_b64=heatmap_b64,
        bboxes=bboxes
    )

# Static file serving
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
# Create dir initially if missing just so mount doesn't crash on boot before index is ready
os.makedirs(FRONTEND_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

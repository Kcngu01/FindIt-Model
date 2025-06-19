import io
import base64
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torchvision.transforms as transforms
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import json
import os
import tempfile
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("findit-api")

# Set custom cache directory with write permissions
cache_dir = os.path.join(tempfile.gettempdir(), "huggingface_cache")
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = os.path.join(tempfile.gettempdir(), "huggingface_home")
os.environ["HF_DATASETS_CACHE"] = os.path.join(tempfile.gettempdir(), "huggingface_datasets")

# Ensure cache directories exist with proper permissions
for dir_path in [cache_dir, os.environ["HF_HOME"], os.environ["HF_DATASETS_CACHE"]]:
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Cache directory created/verified: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating cache directory {dir_path}: {e}")

app = FastAPI(title="Image Similarity API with DINO v2")

# Add CORS middleware to allow requests from Laravel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo purposes, you can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None
model_name = None
device = "cpu"

# Initialize DINO v2 model from Hugging Face
def load_model(model_variant="base"):
    global model, processor, model_name, device
    
    try:
        # Determine which model to load
        model_name = f"facebook/dinov2-{model_variant}"
        logger.info(f"Attempting to load model: {model_name}")
        
        # Log cache directory information
        logger.info(f"Using cache directory: {cache_dir}")
        logger.info(f"Cache directory exists: {os.path.exists(cache_dir)}")
        logger.info(f"Cache directory permissions: {oct(os.stat(cache_dir).st_mode)[-3:]}")
        
        # Load the model with explicit cache directory
        processor = AutoImageProcessor.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        model = AutoModel.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            local_files_only=False
        )

        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        logger.info(f"DINO v2 model loaded successfully: {model_name}. Using device: {device}")
        return True
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return False

# Try to load models in order of preference
if not load_model("base"):
    logger.warning("Failed to load base model, trying small model")
    if not load_model("small"):
        logger.error("Failed to load any DINO v2 model")
        # We'll initialize a dummy model for demonstration purposes
        # In production, you might want to raise an exception here
        class DummyModel:
            def __init__(self):
                self.last_hidden_state = None
                
            def __call__(self, **kwargs):
                # Return a simple tensor with the right shape for demonstration
                batch_size = kwargs.get('pixel_values', torch.zeros(1, 3, 224, 224)).shape[0]
                self.last_hidden_state = torch.zeros(batch_size, 1, 768)
                return self
                
            def to(self, device):
                return self
                
            def eval(self):
                pass
                
        class DummyProcessor:
            def __call__(self, images, return_tensors="pt"):
                if isinstance(images, list):
                    batch_size = len(images)
                else:
                    batch_size = 1
                return {"pixel_values": torch.zeros(batch_size, 3, 224, 224)}
        
        model = DummyModel()
        processor = DummyProcessor()
        model_name = "dummy-model"
        logger.warning("Initialized dummy model for demonstration. API will return random embeddings.")

# Pydantic models for request/response
class ImageBase64(BaseModel):
    image: str

class ItemEmbedding(BaseModel):
    item_id: int
    embedding: List[float]
    category_id: int
    color_id: int
    location_id: int

class SimilarityRequest(BaseModel):
    new_image: str  # Base64 encoded image
    stored_embeddings: List[ItemEmbedding] = []
    threshold: float = 0.7  # Default threshold

class SimilarityMatch(BaseModel):
    item_id: int
    similarity: float

class SimilarityResponse(BaseModel):
    embedding: List[float]
    matches: List[SimilarityMatch] = []

# Helper function to get image embedding
def get_image_embedding(image_data):
    try:
        # Convert base64 to image
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Handle data URL format
            image_data = image_data.split(',')[1]

        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # Handle direct image file upload
            image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if image is not in RGB mode (e.g., RGBA)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image with Hugging Face processor
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token as embedding (first token of last hidden state)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

        # Normalize the embedding
        embedding_norm = F.normalize(torch.tensor(embedding).unsqueeze(0), p=2, dim=1)

        return embedding_norm.squeeze().tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Endpoint to compute image embedding
@app.post("/compute_embedding", response_model=Dict[str, List[float]])
async def compute_embedding(image_data: ImageBase64):
    embedding = get_image_embedding(image_data.image)
    return {"embedding": embedding}

# Endpoint to perform similarity comparison
@app.post("/compare_similarity", response_model=SimilarityResponse)
async def compare_similarity(request: SimilarityRequest):
    # Get embedding for the new image
    new_embedding = get_image_embedding(request.new_image)
    new_embedding_tensor = torch.tensor(new_embedding)

    matches = []

    # Compare with stored embeddings
    for item in request.stored_embeddings:
        stored_embedding = torch.tensor(item.embedding)

        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            new_embedding_tensor.unsqueeze(0),
            stored_embedding.unsqueeze(0)
        ).item()

        # Check if similarity exceeds threshold
        if similarity >= request.threshold:
            matches.append(SimilarityMatch(
                item_id=item.item_id,
                similarity=similarity
            ))

    # Sort matches by similarity (highest first)
    matches.sort(key=lambda x: x.similarity, reverse=True)

    return SimilarityResponse(
        embedding=new_embedding,
        matches=matches
    )

# Endpoint for multipart form data (direct file upload)
@app.post("/compare_similarity_file", response_model=SimilarityResponse)
async def compare_similarity_file(
    image: UploadFile = File(...),
    stored_embeddings: str = Form(...),  # JSON string of embeddings
    threshold: float = Form(0.7)
):
    # Parse the embeddings from JSON
    embeddings_data = json.loads(stored_embeddings)
    stored_items = [
        ItemEmbedding(
            item_id=item["item_id"],
            embedding=item["embedding"],
            category_id=item["category_id"],
            color_id=item["color_id"],
            location_id=item["location_id"]
        )
        for item in embeddings_data
    ]

    # Read image data
    image_data = await image.read()

    # Get embedding for the new image
    new_embedding = get_image_embedding(image_data)
    new_embedding_tensor = torch.tensor(new_embedding)

    matches = []

    # Compare with stored embeddings
    for item in stored_items:
        stored_embedding = torch.tensor(item.embedding)

        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            new_embedding_tensor.unsqueeze(0),
            stored_embedding.unsqueeze(0)
        ).item()

        # Check if similarity exceeds threshold
        if similarity >= threshold:
            matches.append(SimilarityMatch(
                item_id=item.item_id,
                similarity=similarity
            ))

    # Sort matches by similarity (highest first)
    matches.sort(key=lambda x: x.similarity, reverse=True)

    return SimilarityResponse(
        embedding=new_embedding,
        matches=matches
    )

# Simple health check endpoint
@app.get("/")
def read_root():
    return {"status": "ok", "model": "DINO v2 (ViT-B/16)", "message": "Image Similarity API is running on Hugging Face Spaces"}

# Add a health check endpoint for monitoring
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_name": model_name,
        "device": device,
        "torch_version": torch.__version__
    }
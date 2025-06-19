metadata
title: FindIt Api
emoji: 📈
colorFrom: green
colorTo: red
sdk: docker
pinned: false
short_description: Generate Image Imbeddings and Compare Image Similarity
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

DINO v2 Image Similarity API
This is a FastAPI service that provides image similarity comparison using Facebook's DINO v2 model. The API can compute embeddings for images and compare them against stored embeddings to find similar items.

Features
Image Embedding Generation: Convert images to high-dimensional embeddings using DINO v2
Similarity Comparison: Compare new images against stored embeddings
Multiple Input Formats: Support for base64 encoded images and direct file uploads
Configurable Threshold: Adjustable similarity threshold for matches
CORS Enabled: Ready for integration with web applications
API Endpoints
GET /
Health check endpoint that returns the API status and model information.

GET /health
Detailed health check with system information.

POST /compute_embedding
Generate an embedding for a single image.

Request Body:

{
  "image": "base64_encoded_image_string"
}
Response:

{
  "embedding": [0.1, 0.2, 0.3, ...]
}
POST /compare_similarity
Compare a new image against stored embeddings.

Request Body:

{
  "new_image": "base64_encoded_image_string",
  "stored_embeddings": [
    {
      "item_id": 1,
      "embedding": [0.1, 0.2, 0.3, ...],
      "category_id": 1,
      "color_id": 1,
      "location_id": 1
    }
  ],
  "threshold": 0.7
}
Response:

{
  "embedding": [0.1, 0.2, 0.3, ...],
  "matches": [
    {
      "item_id": 1,
      "similarity": 0.85
    }
  ]
}
POST /compare_similarity_file
Compare an uploaded image file against stored embeddings (multipart form data).

Form Data:

image: Image file
stored_embeddings: JSON string of stored embeddings
threshold: Similarity threshold (default: 0.7)
Usage with Laravel
To integrate this API with your Laravel backend, update your HTTP client calls to use the Hugging Face Spaces URL:

$response = Http::post('https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/compare_similarity', [
    'new_image' => $base64Image,
    'stored_embeddings' => $storedEmbeddings,
    'threshold' => 0.7
]);
Model Information
Model: Facebook DINO v2 (ViT-B/16)
Framework: PyTorch + Transformers
Embedding Dimension: 768
Input: RGB images (automatically converted if needed)
Similarity Metric: Cosine similarity
Implementation Notes
The application uses a custom cache directory in the temporary folder to avoid permission issues when downloading models from Hugging Face.
If the primary model fails to load, the application will attempt to load a smaller version of the DINO v2 model.
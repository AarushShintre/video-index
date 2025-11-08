# Semantic Search Setup Guide

This guide explains how to set up and use semantic video search in the VideoHub app.

## Overview

Semantic search allows users to find videos that are visually similar to a selected video, using deep learning embeddings (ResNet-50) and FAISS for fast similarity search.

## Architecture

1. **Python Flask Service** (`server/src/services/semanticSearch.py`): Handles semantic search queries
2. **Node.js API** (`server/src/controllers/semanticSearchController.js`): Bridges the Node.js server and Python service
3. **React Client** (`client/src/App.jsx`): Displays similar videos in the UI

## Prerequisites

1. Python 3.8+ with required packages
2. Node.js server running
3. Video embeddings and FAISS index generated

## Setup Steps

### 1. Install Python Dependencies

```bash
cd server/src/services
pip install -r requirements.txt
```

Or install from the main requirements:
```bash
pip install flask flask-cors numpy faiss-cpu scikit-learn torch torchvision opencv-python
```

### 2. Generate Video Embeddings and Index

First, you need to process your videos to generate embeddings and build the FAISS index:

```bash
# From the project root
python video_clustering.py \
    --max-videos 100 \
    --n-clusters 10 \
    --normalize \
    --use-cosine \
    --pca-components 128 \
    --output-dir output
```

This will create:
- `output/video_embeddings.npz`: Video embeddings
- `output/clustering_results.npz`: Clustering results
- `output/faiss_index.bin`: FAISS index for fast search
- `output/pca_model.pkl`: PCA model for dimensionality reduction

### 3. Start the Python Semantic Search Service

**Windows:**
```bash
cd server/src/services
python semanticSearch.py --port 5001 --results-dir ../../output
```

**Linux/Mac:**
```bash
cd server/src/services
chmod +x start_semantic_search.sh
./start_semantic_search.sh
```

Or manually:
```bash
python semanticSearch.py --port 5001 --results-dir ../../output
```

The service should start on `http://localhost:5001`

### 4. Install Node.js Dependencies

```bash
cd server
npm install
```

This will install `axios` which is needed to call the Python service.

### 5. Start the Node.js Server

```bash
cd server
npm run dev
```

The server should start on `http://localhost:5000`

### 6. Start the React Client

```bash
cd client
npm install
npm run dev
```

## Usage

1. **Upload Videos**: Upload videos through the web interface
2. **View Video**: Click on any video to view it
3. **Find Similar Videos**: When viewing a video, if semantic search is available, you'll see a "Similar Videos" section showing visually similar videos

## How It Works

1. When you click on a video, the client calls `/api/videos/:id/similar`
2. The Node.js server forwards the request to the Python semantic search service
3. The Python service:
   - Extracts embedding from the query video using ResNet-50
   - Applies PCA transformation (if used during indexing)
   - Normalizes the embedding (if normalization was used)
   - Searches the FAISS index for similar videos
   - Returns top-k most similar videos with similarity scores

## Troubleshooting

### "Semantic search service unavailable"

- Make sure the Python service is running on port 5001
- Check that the service can access the `output` directory with embeddings

### "Video file not found"

- Ensure uploaded videos are in `server/uploads/`
- Check that video paths in the database match actual file locations

### "No similar videos found"

- Make sure you've generated embeddings for your videos
- The query video needs to be in the indexed set (or you need to rebuild the index)
- Check that the FAISS index contains videos

### Dimension Mismatch Errors

- Ensure the query video uses the same embedding model as the index
- Make sure PCA model matches the one used during indexing
- Check normalization settings match between indexing and querying

## Adding New Videos to the Index

Currently, adding new videos requires rebuilding the entire index:

1. Extract embeddings for all videos (including new ones)
2. Rebuild the FAISS index
3. Restart the semantic search service

Future enhancement: Implement incremental index updates to add videos without full rebuild.

## Configuration

Set environment variables if needed:

- `SEMANTIC_SEARCH_URL`: URL of the Python service (default: `http://localhost:5001`)
- `UPLOAD_DIR`: Directory where videos are stored (default: `server/uploads`)

## API Endpoints

### Node.js Server

- `POST /api/videos/:id/similar`: Find similar videos
- `GET /api/videos/semantic-search/health`: Check if semantic search is available

### Python Service

- `GET /health`: Health check
- `POST /search`: Search for similar videos
- `POST /add_video`: Add video to index (requires rebuild)

## Notes

- The semantic search uses visual content (frames), not audio or text
- Similarity is based on visual features extracted by ResNet-50
- For better results, use `--normalize` and `--use-cosine` flags when generating embeddings
- The service automatically handles PCA transformation and normalization


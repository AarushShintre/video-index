# VideoHub Semantic Search Integration

## ✅ What Was Added

Semantic search functionality has been integrated into your VideoHub app! Users can now find visually similar videos using deep learning embeddings.

## 🎯 Features

1. **Similar Videos Section**: When viewing a video, a "Similar Videos" section appears showing visually similar videos
2. **Automatic Detection**: The app automatically detects if semantic search is available
3. **Visual Similarity**: Uses ResNet-50 embeddings to find videos with similar visual content
4. **Fast Search**: FAISS index enables fast similarity search

## 📁 Files Added/Modified

### New Files:
- `server/src/services/semanticSearch.py` - Python Flask service for semantic search
- `server/src/controllers/semanticSearchController.js` - Node.js controller for semantic search
- `server/src/services/requirements.txt` - Python dependencies
- `server/src/services/start_semantic_search.sh` - Startup script (Linux/Mac)
- `server/src/services/start_semantic_search.bat` - Startup script (Windows)
- `SEMANTIC_SEARCH_SETUP.md` - Detailed setup guide

### Modified Files:
- `server/src/routes/videos.js` - Added semantic search routes
- `server/package.json` - Added axios dependency
- `client/src/services/api.js` - Added semantic search API methods
- `client/src/App.jsx` - Added similar videos UI section

## 🚀 Quick Start

### 1. Install Dependencies

**Python:**
```bash
cd server/src/services
pip install -r requirements.txt
```

**Node.js:**
```bash
cd server
npm install
```

### 2. Generate Video Embeddings

First, you need to process your videos to create embeddings and the search index:

```bash
# From project root
python video_clustering.py \
    --max-videos 100 \
    --normalize \
    --use-cosine \
    --output-dir output
```

This processes videos and creates:
- Embeddings for all videos
- FAISS index for fast search
- PCA model for dimensionality reduction

### 3. Start Services

**Terminal 1 - Python Semantic Search Service:**
```bash
cd server/src/services
python semanticSearch.py --port 5001 --results-dir ../../output
```

**Terminal 2 - Node.js Server:**
```bash
cd server
npm run dev
```

**Terminal 3 - React Client:**
```bash
cd client
npm run dev
```

### 4. Use It!

1. Open the app in your browser
2. Click on any video
3. Scroll down to see "Similar Videos" section
4. Click on similar videos to navigate

## 🔧 How It Works

1. **User clicks a video** → Client calls `/api/videos/:id/similar`
2. **Node.js server** → Gets video info from database, calls Python service
3. **Python service** → 
   - Extracts embedding from query video (ResNet-50)
   - Applies PCA transformation
   - Searches FAISS index
   - Returns top-k similar videos
4. **Client** → Displays similar videos with similarity scores

## ⚠️ Important Notes

### Video Indexing

- Videos need to be indexed **before** they can be found in semantic search
- Currently, you need to run `video_clustering.py` to index videos
- The indexed videos must match the videos in your database

### Matching Videos

The semantic search returns video file paths. The client tries to match these to videos in your database by filename. Make sure:
- Video filenames in the index match filenames in the database
- Videos are in the `server/uploads/` directory

### Adding New Videos

To add new videos to the semantic search:
1. Upload videos through the web interface
2. Run `video_clustering.py` again to rebuild the index with new videos
3. Restart the Python semantic search service

## 🐛 Troubleshooting

### "Semantic search service unavailable"
- Check if Python service is running on port 5001
- Verify the service can access the `output` directory

### "No similar videos found"
- Make sure videos are indexed (run `video_clustering.py`)
- Check that video filenames match between index and database

### Similar videos don't match database videos
- Ensure filenames in the FAISS index match `filepath` in the database
- Check that videos are in `server/uploads/` directory

## 📊 Performance

- **Embedding Extraction**: ~1-2 seconds per video (CPU)
- **Similarity Search**: <100ms for queries (FAISS)
- **Index Size**: Depends on number of videos and embedding dimensions

## 🔮 Future Enhancements

- Incremental index updates (add videos without full rebuild)
- Real-time embedding extraction on upload
- Better video matching (by ID instead of filename)
- Support for video-based temporal models (I3D, TimeSformer)

## 📚 More Information

See `SEMANTIC_SEARCH_SETUP.md` for detailed setup instructions and API documentation.


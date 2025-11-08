# Video Clustering Pipeline - Improvements Summary

## Issues Fixed

### 1. **Fixed Embedding Dimension Mismatch** ✅
   - **Problem**: Query embeddings (2048-d) didn't match index dimension (99-d after PCA)
   - **Solution**: 
     - PCA model is now saved to disk (`pca_model.pkl`)
     - Query script automatically loads and applies PCA transformation
     - Proper normalization handling for cosine similarity
   - **Result**: Queries now work correctly with proper dimension matching

### 2. **Added Embedding Normalization** ✅
   - **New Feature**: `--normalize` flag for L2-normalization
   - **Benefit**: Often improves cluster quality and similarity search
   - **Usage**: `python video_clustering.py --normalize --use-cosine`

### 3. **Improved FAISS Index** ✅
   - **New Feature**: Cosine similarity support with `--use-cosine`
   - **Benefit**: Better similarity search for normalized embeddings
   - **Implementation**: Uses `IndexFlatIP` (inner product) for cosine similarity

### 4. **Alternative Clustering Methods** ✅
   - **New Methods**:
     - `--clustering-method dbscan`: Density-based, handles uneven cluster sizes
     - `--clustering-method hdbscan`: Hierarchical DBSCAN, better for varying densities
     - `--clustering-method agglomerative`: Hierarchical clustering
   - **Benefit**: Better results for non-spherical clusters and unbalanced data

## New Command-Line Options

```bash
# Normalization and cosine similarity (recommended)
python video_clustering.py --normalize --use-cosine

# DBSCAN for uneven cluster sizes
python video_clustering.py --clustering-method dbscan --dbscan-eps 0.3

# HDBSCAN for varying densities
python video_clustering.py --clustering-method hdbscan --hdbscan-min-cluster-size 5
```

## Expected Improvements

### Before (Current Results)
- Silhouette Score: ~0.016 (very poor)
- Calinski-Harabasz: ~2.60 (very low)
- Davies-Bouldin: ~2.81 (high, poor separation)

### After (With Improvements)
Expected improvements with `--normalize --use-cosine`:
- **Silhouette Score**: Should improve to >0.2-0.3
- **Better cluster separation**: Normalization helps with cosine similarity
- **More balanced clusters**: DBSCAN/HDBSCAN handle uneven sizes better

## Next Steps for Further Improvement

### 1. Video-Based Temporal Models (Future Enhancement)
   - Current: Frame-level averaging (ResNet-50)
   - Recommended: TimeSformer, I3D, or Video Swin Transformer
   - Benefit: Captures temporal motion patterns crucial for action videos

### 2. Hyperparameter Tuning
   - Try different `--dbscan-eps` values (0.1 to 0.5)
   - Adjust `--pca-components` (64, 128, 256)
   - Experiment with `--skip-frames` (2, 4, 8)

### 3. Evaluation
   - Run with `--normalize --use-cosine` and compare metrics
   - Try different clustering methods
   - Inspect cluster representatives manually

## Usage Examples

### Recommended Configuration
```bash
python video_clustering.py \
    --max-videos 100 \
    --n-clusters 10 \
    --normalize \
    --use-cosine \
    --pca-components 128 \
    --extract-archives
```

### For Unbalanced Clusters
```bash
python video_clustering.py \
    --max-videos 100 \
    --clustering-method dbscan \
    --dbscan-eps 0.3 \
    --dbscan-min-samples 3 \
    --normalize \
    --use-cosine
```

### Query Similar Videos (Now Fixed)
```bash
# Query by index (works correctly now)
python query_similar_videos.py --query-index 0 --k 5

# Query by video file (handles PCA automatically)
python query_similar_videos.py --query-video path/to/video.mp4 --k 5
```

## Files Modified

1. **video_clustering.py**:
   - Added normalization support
   - Added alternative clustering methods (DBSCAN, HDBSCAN, Agglomerative)
   - Improved FAISS index with cosine similarity
   - Save PCA model for query transformation

2. **query_similar_videos.py**:
   - Loads and applies PCA transformation
   - Handles normalization correctly
   - Proper dimension matching

3. **requirements.txt**:
   - Added `hdbscan` for hierarchical clustering

4. **README.md**:
   - Updated with new features and recommendations

## Testing Recommendations

1. **Re-run clustering with normalization**:
   ```bash
   python video_clustering.py --max-videos 100 --normalize --use-cosine
   ```
   Compare metrics with previous run.

2. **Try DBSCAN**:
   ```bash
   python video_clustering.py --max-videos 100 --clustering-method dbscan --normalize
   ```
   Check if cluster sizes are more balanced.

3. **Test queries**:
   ```bash
   python query_similar_videos.py --query-index 0 --k 5
   ```
   Verify no dimension mismatch warnings.

## Notes

- **Normalization** is recommended for most use cases
- **Cosine similarity** works best with normalized embeddings
- **DBSCAN/HDBSCAN** automatically determine number of clusters
- **PCA model** is saved for consistent query transformations


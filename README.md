# Video Clustering Pipeline

A comprehensive pipeline for clustering videos from the Something-Something dataset based on visual content using deep learning embeddings.

## Features

- **Frame-level Embedding Extraction**: Uses pre-trained ResNet-50 to extract visual features
- **Dimensionality Reduction**: Optional PCA for noise reduction and faster processing
- **Clustering**: K-Means clustering for grouping similar videos
- **Cluster Quality Evaluation**: Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index
- **Cluster Representatives**: Automatically finds representative videos for each cluster
- **Similarity Search**: FAISS-based fast similarity search with query interface
- **Visualization**: UMAP-based 2D visualization of clusters

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

**Note**: For GPU acceleration, install `faiss-gpu` instead of `faiss-cpu`:
```bash
pip install faiss-gpu
```

## Usage

### Basic Usage

Process videos in the current directory and cluster them:

```bash
python video_clustering.py
```

### Advanced Options

```bash
python video_clustering.py \
    --data-dir . \
    --output-dir output \
    --n-clusters 10 \
    --max-videos 1000 \
    --skip-frames 4 \
    --pca-components 128 \
    --extract-archives
```

### Command Line Arguments

- `--data-dir`: Directory containing video files or archives (default: current directory)
- `--output-dir`: Output directory for results (default: `output`)
- `--n-clusters`: Number of clusters for K-Means (default: 10)
- `--max-videos`: Maximum number of videos to process (default: all)
- `--skip-frames`: Number of frames to skip between samples (default: 4)
- `--pca-components`: Number of PCA components for dimensionality reduction (default: 128)
- `--no-pca`: Disable PCA dimensionality reduction
- `--extract-archives`: Extract videos from tar archives (if archives are detected)
- `--load-embeddings`: Load pre-computed embeddings from file (saves time on re-runs)
- `--normalize`: L2-normalize embeddings before clustering (recommended for better results)
- `--clustering-method`: Clustering algorithm - `kmeans`, `dbscan`, `hdbscan`, or `agglomerative` (default: `kmeans`)
- `--dbscan-eps`: DBSCAN eps parameter (default: 0.5)
- `--dbscan-min-samples`: DBSCAN min_samples parameter (default: 5)
- `--hdbscan-min-cluster-size`: HDBSCAN min_cluster_size parameter (default: 5)
- `--use-cosine`: Use cosine similarity for FAISS index (requires `--normalize`)

### Example Workflows

#### 1. Process a subset of videos for testing

```bash
python video_clustering.py --max-videos 100 --n-clusters 5
```

#### 2. Process all videos with custom settings

```bash
python video_clustering.py \
    --n-clusters 20 \
    --skip-frames 8 \
    --pca-components 256 \
    --output-dir results
```

#### 3. Load pre-computed embeddings (faster re-clustering)

```bash
# First run: extract embeddings
python video_clustering.py --max-videos 500

# Second run: use saved embeddings with different cluster count
python video_clustering.py \
    --load-embeddings output/video_embeddings.npz \
    --n-clusters 15
```

### Cluster Quality Evaluation

The pipeline automatically evaluates cluster quality using multiple metrics:

- **Silhouette Score**: Measures how well videos are separated into clusters (range: -1 to 1, higher is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance (higher is better)
- **Davies-Bouldin Index**: Average similarity ratio of clusters (lower is better)

You can also evaluate clusters separately:

```bash
python evaluate_clusters.py --results-file output/clustering_results.npz
```

### Similarity Search

Query for similar videos using the pre-built FAISS index:

#### Query by video index (from dataset)

```bash
python query_similar_videos.py --query-index 0 --k 5
```

#### Query by video file path

```bash
python query_similar_videos.py --query-video path/to/video.mp4 --k 5
```

This is useful for:
- **Content-based recommendation**: Find videos similar to a given video
- **Duplicate detection**: Identify near-duplicate videos
- **Cluster exploration**: Understand what makes videos similar

## Output Files

The pipeline generates several output files in the `output` directory:

- `video_embeddings.npz`: Saved video embeddings and file paths
- `clustering_results.npz`: Cluster labels, reduced embeddings, and quality metrics
- `faiss_index.bin`: FAISS index for fast similarity search
- `cluster_visualization.png`: 2D UMAP visualization of clusters

## How It Works

1. **Video Loading**: Finds video files in the specified directory or extracts from archives
2. **Feature Extraction**: 
   - Samples frames from each video (with configurable skip rate)
   - Extracts embeddings using ResNet-50 (pre-trained on ImageNet)
   - Averages frame embeddings to get a single vector per video
3. **Dimensionality Reduction**: Optional PCA to reduce noise and speed up clustering
4. **Clustering**: K-Means clustering groups similar videos
5. **Quality Evaluation**: Computes multiple metrics to assess cluster quality
6. **Representative Selection**: Finds videos closest to cluster centroids
7. **Similarity Search**: FAISS index enables fast nearest neighbor queries
8. **Visualization**: UMAP projects high-dimensional embeddings to 2D for visualization

## Performance Tips

- **GPU Acceleration**: The pipeline automatically uses GPU if available for faster embedding extraction
- **Batch Processing**: For large datasets, process in batches using `--max-videos`
- **Skip Frames**: Increase `--skip-frames` for faster processing (at cost of some temporal information)
- **PCA**: Use PCA to reduce dimensionality and speed up clustering (recommended for >1000 videos)

## Scaling Up

For larger datasets:

1. **Distributed Processing**: Process videos in parallel across multiple machines
2. **Vector Database**: Use Milvus or Pinecone for millions of vectors
3. **Video Models**: Consider I3D, C3D, or TimeSformer for temporal embeddings (better for action recognition)
4. **Cloud Storage**: Store videos in S3/GCS and process in batches

## Cluster Quality Interpretation

### Good Clusters
- **Silhouette Score > 0.3**: Generally indicates reasonable cluster separation
- **Calinski-Harabasz Index**: Higher values indicate better-defined clusters
- **Davies-Bouldin Index < 1.0**: Lower values indicate better cluster separation

### Improving Cluster Quality

If clusters are not meaningful:

1. **Enable normalization**: Use `--normalize` flag to L2-normalize embeddings (often improves results)
2. **Use cosine similarity**: Combine `--normalize` and `--use-cosine` for better similarity search
3. **Try alternative clustering methods**:
   - `--clustering-method dbscan`: Good for uneven cluster sizes, automatically finds number of clusters
   - `--clustering-method hdbscan`: Hierarchical DBSCAN, handles varying density
   - `--clustering-method agglomerative`: Hierarchical clustering with specified number of clusters
4. **Adjust number of clusters**: Try different `--n-clusters` values
5. **Increase embedding quality**: 
   - Reduce `--skip-frames` to capture more temporal information
   - Use video-based models (I3D, TimeSformer) instead of frame-level (future enhancement)
6. **Manual inspection**: Review cluster representatives to understand what's being grouped

### Recommended Settings for Better Results

For improved cluster quality, try:

```bash
python video_clustering.py \
    --max-videos 100 \
    --n-clusters 10 \
    --normalize \
    --use-cosine \
    --pca-components 128
```

Or for uneven cluster sizes:

```bash
python video_clustering.py \
    --max-videos 100 \
    --clustering-method dbscan \
    --dbscan-eps 0.3 \
    --dbscan-min-samples 3 \
    --normalize \
    --use-cosine
```

## Troubleshooting

- **Out of Memory**: Reduce `--max-videos` or increase `--skip-frames`
- **Slow Processing**: Enable GPU or reduce number of videos
- **No Videos Found**: Check `--data-dir` path and ensure videos are in supported formats (.mp4, .avi, .mov, .webm)
- **Split Archive Errors**: If you have files like `20bn-something-something-v2-00` and `20bn-something-something-v2-01`, these are split archives. The script will automatically detect and concatenate them when using `--extract-archives`. If extraction fails, the files might be:
  - Raw video files (try without `--extract-archives`)
  - A different archive format (may need manual extraction)
  - Corrupted or incomplete downloads
- **Low Cluster Quality Scores**: 
  - Enable `--normalize` and `--use-cosine` for better similarity
  - Try `--clustering-method dbscan` or `hdbscan` for uneven cluster sizes
  - Adjust `--n-clusters`, reduce `--skip-frames`, or disable PCA with `--no-pca`
- **Dimension Mismatch in Queries**: The query script now automatically handles PCA transformation. Make sure you're using the same embedding model as during clustering.

## License

This project is provided as-is for research and educational purposes.


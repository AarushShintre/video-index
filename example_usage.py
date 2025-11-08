"""
Example usage of the video clustering pipeline.
This script demonstrates how to use the pipeline programmatically.
"""

from video_clustering import (
    VideoEmbeddingExtractor,
    find_video_files,
    process_videos,
    cluster_videos,
    build_faiss_index,
    visualize_clusters,
    print_cluster_summary
)
import os


def example_basic_clustering():
    """Basic example: cluster videos in a directory."""
    
    # Initialize extractor
    extractor = VideoEmbeddingExtractor(skip_frames=4)
    print(f"Using device: {extractor.device}")
    
    # Find video files
    video_dir = "."  # Change to your video directory
    video_files = find_video_files(video_dir)
    
    if len(video_files) == 0:
        print("No video files found!")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Limit to first 50 videos for quick testing
    video_files = video_files[:50]
    
    # Extract embeddings
    video_embeddings, valid_video_files = process_videos(
        video_files, 
        extractor,
        output_file="embeddings.npz"
    )
    
    # Cluster videos
    labels, embeddings_reduced, pca = cluster_videos(
        video_embeddings,
        n_clusters=5,
        use_pca=True,
        pca_components=128
    )
    
    # Print summary
    print_cluster_summary(labels, valid_video_files, n_clusters=5)
    
    # Build FAISS index for similarity search
    faiss_index = build_faiss_index(embeddings_reduced)
    
    # Find similar videos
    if len(embeddings_reduced) > 0:
        query_idx = 0
        query = embeddings_reduced[query_idx:query_idx+1].astype('float32')
        k = min(5, len(embeddings_reduced))
        D, I = faiss_index.search(query, k)
        
        print(f"\nVideos similar to: {os.path.basename(valid_video_files[query_idx])}")
        for idx, dist in zip(I[0][1:], D[0][1:]):  # Skip first (same video)
            print(f"  - {os.path.basename(valid_video_files[idx])} (distance: {dist:.2f})")
    
    # Visualize
    visualize_clusters(embeddings_reduced, labels, "example_clusters.png")


def example_load_and_recluster():
    """Example: Load pre-computed embeddings and re-cluster with different settings."""
    
    import numpy as np
    
    # Load saved embeddings
    if not os.path.exists("embeddings.npz"):
        print("No saved embeddings found. Run example_basic_clustering() first.")
        return
    
    data = np.load("embeddings.npz", allow_pickle=True)
    video_embeddings = data['embeddings']
    video_files = data['video_files'].tolist()
    
    print(f"Loaded {len(video_embeddings)} embeddings")
    
    # Re-cluster with different number of clusters
    labels, embeddings_reduced, pca = cluster_videos(
        video_embeddings,
        n_clusters=10,  # Different number of clusters
        use_pca=True,
        pca_components=128
    )
    
    # Visualize new clustering
    visualize_clusters(embeddings_reduced, labels, "example_clusters_10.png")


if __name__ == "__main__":
    print("Running basic clustering example...")
    example_basic_clustering()
    
    # Uncomment to test loading and re-clustering:
    # print("\nRunning re-clustering example...")
    # example_load_and_recluster()


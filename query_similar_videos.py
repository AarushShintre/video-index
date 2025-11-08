"""
Query script for finding similar videos using pre-computed embeddings and FAISS index.
"""

import os
import numpy as np
import faiss
import argparse
from video_clustering import (
    VideoEmbeddingExtractor,
    find_similar_videos,
    build_faiss_index
)


def load_index_and_data(results_dir):
    """
    Load FAISS index, clustering results, and PCA model.
    
    Args:
        results_dir: Directory containing clustering results
        
    Returns:
        tuple: (faiss_index, embeddings, video_files, pca_model, normalize, use_cosine)
    """
    import pickle
    
    # Load clustering results
    results_file = os.path.join(results_dir, "clustering_results.npz")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    data = np.load(results_file, allow_pickle=True)
    embeddings = data['embeddings']
    video_files = data['video_files'].tolist()
    normalize = bool(data.get('normalize', False))
    use_cosine = bool(data.get('use_cosine', False))
    
    # Load PCA model if it exists
    pca_file = os.path.join(results_dir, "pca_model.pkl")
    pca_model = None
    if os.path.exists(pca_file):
        with open(pca_file, 'rb') as f:
            pca_model = pickle.load(f)
        print(f"Loaded PCA model from {pca_file}")
    
    # Try to load FAISS index
    faiss_index_file = os.path.join(results_dir, "faiss_index.bin")
    if os.path.exists(faiss_index_file):
        print(f"Loading FAISS index from {faiss_index_file}...")
        faiss_index = faiss.read_index(faiss_index_file)
    else:
        # Build index from embeddings if not found
        print("FAISS index not found. Building from embeddings...")
        from video_clustering import build_faiss_index
        faiss_index = build_faiss_index(embeddings, use_cosine=use_cosine)
    
    return faiss_index, embeddings, video_files, pca_model, normalize, use_cosine


def query_by_video_path(query_video_path, results_dir, k=5, extractor=None):
    """
    Find similar videos by querying with a video file path.
    
    Args:
        query_video_path: Path to query video
        results_dir: Directory containing clustering results
        k: Number of similar videos to return
        extractor: VideoEmbeddingExtractor instance (optional)
        
    Returns:
        list: List of tuples (video_path, distance)
    """
    from sklearn.preprocessing import normalize as sklearn_normalize
    import faiss
    
    if extractor is None:
        extractor = VideoEmbeddingExtractor()
    
    # Extract embedding for query video
    print(f"Extracting embedding from query video: {query_video_path}")
    query_embedding = extractor.extract_video_embedding(query_video_path)
    
    if query_embedding is None:
        raise ValueError(f"Could not extract embedding from {query_video_path}")
    
    # Load index and data
    faiss_index, embeddings, video_files, pca_model, normalize, use_cosine = load_index_and_data(results_dir)
    
    # Normalize if the index was built with normalization
    if normalize:
        print("Normalizing query embedding...")
        query_embedding = sklearn_normalize(query_embedding.reshape(1, -1), norm='l2').flatten()
    
    # Apply PCA transformation if PCA was used
    if pca_model is not None:
        print(f"Applying PCA transformation: {query_embedding.shape[0]} -> {pca_model.n_components} dimensions")
        query_embedding = pca_model.transform(query_embedding.reshape(1, -1)).flatten()
    
    # Verify dimension match
    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError(
            f"Dimension mismatch: Query embedding ({query_embedding.shape[0]}) doesn't match "
            f"index dimension ({embeddings.shape[1]}). "
            f"Make sure you're using the same embedding model and preprocessing."
        )
    
    # Normalize for cosine similarity if needed
    if use_cosine:
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        query_embedding = query_embedding.flatten()
    
    # Find similar videos
    similar_videos = find_similar_videos(query_embedding, faiss_index, video_files, k=k)
    
    return similar_videos


def query_by_index(video_index, results_dir, k=5):
    """
    Find similar videos by querying with a video index from the dataset.
    
    Args:
        video_index: Index of video in the dataset
        results_dir: Directory containing clustering results
        k: Number of similar videos to return
        
    Returns:
        list: List of tuples (video_path, distance)
    """
    # Load index and data
    faiss_index, embeddings, video_files, pca_model, normalize, use_cosine = load_index_and_data(results_dir)
    
    if video_index >= len(embeddings):
        raise ValueError(f"Video index {video_index} out of range (max: {len(embeddings)-1})")
    
    # Use embedding at video_index as query
    query_embedding = embeddings[video_index]
    
    # Find similar videos
    similar_videos = find_similar_videos(query_embedding, faiss_index, video_files, k=k+1)
    
    # Remove the query video itself (first result)
    return similar_videos[1:]


def main():
    parser = argparse.ArgumentParser(description="Query similar videos")
    parser.add_argument("--results-dir", type=str, default="output",
                       help="Directory containing clustering results")
    parser.add_argument("--query-video", type=str, default=None,
                       help="Path to query video file")
    parser.add_argument("--query-index", type=int, default=None,
                       help="Index of video in dataset to use as query")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of similar videos to return")
    
    args = parser.parse_args()
    
    if args.query_video:
        # Query by video file path
        if not os.path.exists(args.query_video):
            print(f"Error: Query video not found: {args.query_video}")
            return
        
        similar_videos = query_by_video_path(
            args.query_video, 
            args.results_dir, 
            k=args.k
        )
        
        print("\n" + "="*60)
        print("SIMILAR VIDEOS")
        print("="*60)
        print(f"\nQuery video: {os.path.basename(args.query_video)}")
        print(f"\nTop {len(similar_videos)} most similar videos:")
        for i, (vid_path, dist) in enumerate(similar_videos, 1):
            print(f"{i}. {os.path.basename(vid_path)} (distance: {dist:.4f})")
    
    elif args.query_index is not None:
        # Query by video index
        similar_videos = query_by_index(
            args.query_index,
            args.results_dir,
            k=args.k
        )
        
        # Load video files to get query video name
        results_file = os.path.join(args.results_dir, "clustering_results.npz")
        data = np.load(results_file, allow_pickle=True)
        video_files = data['video_files'].tolist()
        
        print("\n" + "="*60)
        print("SIMILAR VIDEOS")
        print("="*60)
        print(f"\nQuery video (index {args.query_index}): {os.path.basename(video_files[args.query_index])}")
        print(f"\nTop {len(similar_videos)} most similar videos:")
        for i, (vid_path, dist) in enumerate(similar_videos, 1):
            print(f"{i}. {os.path.basename(vid_path)} (distance: {dist:.4f})")
    
    else:
        print("Error: Must specify either --query-video or --query-index")
        parser.print_help()


if __name__ == "__main__":
    main()


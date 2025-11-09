"""
Python helper script for semantic video search.
Called from Express.js service to handle embedding extraction and FAISS operations.
"""

import sys
import os
import json
import numpy as np
import faiss
import pickle
from pathlib import Path

# Add parent directory to path to import video_clustering
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from video_clustering import VideoEmbeddingExtractor

# Global variables
faiss_index = None
pca_model = None
normalize_embeddings = False
use_cosine = False
embeddings_dir = None
extractor = None
video_files = []


def load_models(results_dir):
    """Load FAISS index, PCA model, and video mappings."""
    global faiss_index, pca_model, normalize_embeddings, use_cosine, embeddings_dir, video_files
    
    embeddings_dir = results_dir
    
    # Load clustering results
    results_file = os.path.join(results_dir, 'clustering_results.npz')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    data = np.load(results_file, allow_pickle=True)
    normalize_embeddings = bool(data.get('normalize', False))
    use_cosine = bool(data.get('use_cosine', False))
    video_files = data['video_files'].tolist()
    
    # Load PCA model if it exists
    pca_file = os.path.join(results_dir, 'pca_model.pkl')
    if os.path.exists(pca_file):
        with open(pca_file, 'rb') as f:
            pca_model = pickle.load(f)
    
    # Load FAISS index
    faiss_index_file = os.path.join(results_dir, 'faiss_index.bin')
    if not os.path.exists(faiss_index_file):
        raise FileNotFoundError(f"FAISS index not found: {faiss_index_file}")
    
    faiss_index = faiss.read_index(faiss_index_file)
    return True


def get_model_info(results_dir):
    """Get information about loaded models."""
    try:
        load_models(results_dir)
        return {
            'index_size': faiss_index.ntotal if faiss_index else 0,
            'normalize_embeddings': normalize_embeddings,
            'use_cosine': use_cosine,
            'has_pca': pca_model is not None
        }
    except Exception as e:
        return {'error': str(e)}


def extract_embedding(video_path):
    """Extract embedding from a video file."""
    global extractor
    
    if extractor is None:
        extractor = VideoEmbeddingExtractor()
    
    embedding = extractor.extract_video_embedding(video_path)
    if embedding is None:
        raise ValueError(f"Failed to extract embedding from {video_path}")
    
    return embedding


def search_similar(video_path, k=5):
    """Find similar videos to a query video."""
    global faiss_index, pca_model, normalize_embeddings, use_cosine, extractor, video_files
    
    if faiss_index is None:
        raise RuntimeError("FAISS index not loaded")
    
    # Extract embedding from query video
    query_embedding = extract_embedding(video_path)
    
    # Normalize if needed
    if normalize_embeddings:
        from sklearn.preprocessing import normalize as sklearn_normalize
        query_embedding = sklearn_normalize(query_embedding.reshape(1, -1), norm='l2').flatten()
    
    # Apply PCA transformation if PCA was used
    if pca_model is not None:
        query_embedding = pca_model.transform(query_embedding.reshape(1, -1)).flatten()
    
    # Normalize for cosine similarity if needed
    if use_cosine:
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        query_embedding = query_embedding.flatten()
    
    # Find similar videos
    query = query_embedding.reshape(1, -1).astype('float32')
    k = min(k, faiss_index.ntotal)
    D, I = faiss_index.search(query, k)
    
    # Return results
    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx < len(video_files):
            video_path_result = video_files[idx]
            results.append({
                'video_path': video_path_result,
                'distance': float(dist),
                'similarity': float(1 / (1 + dist)) if not use_cosine else float(dist)
            })
    
    return {
        'results': results,
        'count': len(results)
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Semantic Search Helper')
    parser.add_argument('--operation', type=str, help='Operation to perform: info, search, extract_embedding')
    parser.add_argument('--results-dir', type=str, help='Directory with clustering results')
    
    args = parser.parse_args()
    
    if not args.results_dir:
        # Try to find results directory
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / 'output',
            Path(__file__).parent / 'output',
            Path.cwd() / 'output'
        ]
        
        for path in possible_paths:
            if path.exists() and (path / 'clustering_results.npz').exists():
                args.results_dir = str(path)
                break
        
        if not args.results_dir:
            print(json.dumps({'error': 'Results directory not found'}), file=sys.stderr)
            sys.exit(1)
    
    try:
        if args.operation == 'info':
            info = get_model_info(args.results_dir)
            print(json.dumps(info))
        
        elif args.operation == 'search':
            load_models(args.results_dir)
            # Read JSON data from stdin to avoid command-line argument parsing issues
            stdin_data = sys.stdin.read()
            if not stdin_data:
                print(json.dumps({'error': 'No data provided via stdin'}), file=sys.stderr)
                sys.exit(1)
            data = json.loads(stdin_data)
            result = search_similar(
                video_path=data['video_path'],
                k=data.get('k', 5)
            )
            print(json.dumps(result))
        
        elif args.operation == 'extract_embedding':
            load_models(args.results_dir)
            # Read JSON data from stdin to avoid command-line argument parsing issues
            stdin_data = sys.stdin.read()
            if not stdin_data:
                print(json.dumps({'error': 'No data provided via stdin'}), file=sys.stderr)
                sys.exit(1)
            data = json.loads(stdin_data)
            embedding = extract_embedding(data['video_path'])
            print(json.dumps({
                'embedding_shape': list(embedding.shape)
            }))
        
        else:
            print(json.dumps({'error': f'Unknown operation: {args.operation}'}), file=sys.stderr)
            sys.exit(1)
    
    except Exception as e:
        import traceback
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_info), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()


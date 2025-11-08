"""
Flask service for semantic video search using FAISS and embeddings.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import faiss
import pickle
from pathlib import Path
import sys

# Add parent directory to path to import video_clustering
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from video_clustering import VideoEmbeddingExtractor, find_similar_videos

app = Flask(__name__)
CORS(app)

# Global variables for loaded models
faiss_index = None
video_files_map = {}  # Maps video_id to file path
pca_model = None
normalize_embeddings = False
use_cosine = False
embeddings_dir = None
extractor = None


def load_semantic_search_models(results_dir='output'):
    """Load FAISS index, PCA model, and video mappings."""
    global faiss_index, pca_model, normalize_embeddings, use_cosine, embeddings_dir
    
    embeddings_dir = results_dir
    
    # Load clustering results
    results_file = os.path.join(results_dir, 'clustering_results.npz')
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found: {results_file}")
        return False
    
    data = np.load(results_file, allow_pickle=True)
    normalize_embeddings = bool(data.get('normalize', False))
    use_cosine = bool(data.get('use_cosine', False))
    
    # Load PCA model if it exists
    pca_file = os.path.join(results_dir, 'pca_model.pkl')
    if os.path.exists(pca_file):
        with open(pca_file, 'rb') as f:
            pca_model = pickle.load(f)
        print(f"Loaded PCA model from {pca_file}")
    
    # Load FAISS index
    faiss_index_file = os.path.join(results_dir, 'faiss_index.bin')
    if os.path.exists(faiss_index_file):
        faiss_index = faiss.read_index(faiss_index_file)
        print(f"Loaded FAISS index with {faiss_index.ntotal} vectors")
    else:
        print(f"Warning: FAISS index not found: {faiss_index_file}")
        return False
    
    return True


def initialize_extractor():
    """Initialize video embedding extractor."""
    global extractor
    if extractor is None:
        extractor = VideoEmbeddingExtractor()
    return extractor


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'index_loaded': faiss_index is not None,
        'index_size': faiss_index.ntotal if faiss_index else 0
    })


@app.route('/routes', methods=['GET'])
def list_routes():
    """List all registered routes for debugging."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify({'routes': routes})


@app.route('/search', methods=['POST'])
@app.route('/search/', methods=['POST'])  # Also handle trailing slash
def search_similar():
    """Find similar videos to a query video."""
    global faiss_index, pca_model, normalize_embeddings, use_cosine, extractor
    
    print(f"DEBUG: /search endpoint called. Method: {request.method}, Path: {request.path}")
    
    if faiss_index is None:
        return jsonify({'error': 'Semantic search index not loaded'}), 500
    
    try:
        data = request.json
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        video_path = data.get('video_path')
        video_id = data.get('video_id')
        k = data.get('k', 5)
        
        if not video_path and not video_id:
            return jsonify({'error': 'Either video_path or video_id required'}), 400
        
        # If video_id provided, try to find in video_files_map
        if video_id and video_id in video_files_map:
            video_path = video_files_map[video_id]
        
        # Handle URL paths - extract filename if it's a URL
        if video_path and ('http://' in video_path or 'https://' in video_path):
            # Extract filename from URL like http://localhost:5000/uploads/filename.mov
            filename = video_path.split('/')[-1]
            # Try to find in uploads directory (relative to server root)
            server_root = Path(__file__).parent.parent.parent.parent
            uploads_dir = server_root / 'server' / 'uploads'
            video_path = str(uploads_dir / filename)
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': f'Video file not found: {video_path}'}), 404
        
        # Initialize extractor if needed
        if extractor is None:
            extractor = initialize_extractor()
        
        # Extract embedding from query video
        query_embedding = extractor.extract_video_embedding(video_path)
        if query_embedding is None:
            return jsonify({'error': 'Failed to extract embedding from video'}), 500
        
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
        
        # Load video files mapping
        results_file = os.path.join(embeddings_dir, 'clustering_results.npz')
        data = np.load(results_file, allow_pickle=True)
        video_files = data['video_files'].tolist()
        
        # Return results
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(video_files):
                video_path = video_files[idx]
                results.append({
                    'video_path': video_path,
                    'distance': float(dist),
                    'similarity': float(1 / (1 + dist)) if not use_cosine else float(dist)  # Convert distance to similarity
                })
        
        return jsonify({
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/add_video', methods=['POST'])
def add_video():
    """Add a new video to the semantic search index."""
    global faiss_index, video_files_map, extractor
    
    if faiss_index is None:
        return jsonify({'error': 'Semantic search index not loaded'}), 500
    
    try:
        data = request.json
        video_path = data.get('video_path')
        video_id = data.get('video_id')
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Initialize extractor if needed
        if extractor is None:
            extractor = initialize_extractor()
        
        # Extract embedding
        embedding = extractor.extract_video_embedding(video_path)
        if embedding is None:
            return jsonify({'error': 'Failed to extract embedding'}), 500
        
        # Apply same transformations as during indexing
        # (This should match the preprocessing used in video_clustering.py)
        # For now, we'll need to save the embedding and rebuild the index
        # In production, you'd want to add it incrementally to FAISS
        
        # Store mapping
        if video_id:
            video_files_map[video_id] = video_path
        
        return jsonify({
            'message': 'Video embedding extracted (index update requires rebuild)',
            'embedding_shape': embedding.shape
        })
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors and show available routes."""
    available_routes = []
    for rule in app.url_map.iter_rules():
        available_routes.append({
            'path': str(rule),
            'methods': list(rule.methods)
        })
    return jsonify({
        'error': 'Route not found',
        'requested_path': request.path,
        'requested_method': request.method,
        'available_routes': available_routes
    }), 404


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Semantic Search Service')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the service on')
    parser.add_argument('--results-dir', type=str, default=None, help='Directory with clustering results')
    
    args = parser.parse_args()
    
    # If no results_dir specified, try to find it relative to project root
    if args.results_dir is None:
        # Try different possible locations
        possible_paths = [
            '../../../output',  # From server/src/services to project root
            '../../output',     # From server/src/services to server/output
            'output',           # Current directory
            Path(__file__).parent.parent.parent.parent / 'output'  # Absolute path to project root
        ]
        
        for path in possible_paths:
            if isinstance(path, Path):
                results_path = path
            else:
                results_path = Path(__file__).parent / path
            results_path = results_path.resolve()
            if results_path.exists() and (results_path / 'clustering_results.npz').exists():
                args.results_dir = str(results_path)
                print(f"Found results directory: {args.results_dir}")
                break
        else:
            args.results_dir = '../../../output'  # Default fallback
            print(f"Using default results directory: {args.results_dir}")
    else:
        # Resolve relative path
        args.results_dir = str(Path(args.results_dir).resolve())
    
    print(f"Loading models from: {args.results_dir}")
    
    # Load models
    if load_semantic_search_models(args.results_dir):
        # Print registered routes for debugging
        print("\n📋 Registered routes:")
        for rule in app.url_map.iter_rules():
            methods = ', '.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
            print(f"  {methods:20} {rule}")
        print()
        
        print(f"✅ Semantic search service ready on port {args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=True)
    else:
        print("❌ Failed to load semantic search models")
        print(f"Checked directory: {args.results_dir}")
        print("Make sure you've run video_clustering.py first to generate embeddings and index")
        print("Or specify the correct --results-dir path")


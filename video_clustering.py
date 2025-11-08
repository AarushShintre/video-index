"""
Video Clustering Pipeline for Something-Something Dataset
Extracts embeddings, clusters videos, and visualizes results.
"""

import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import faiss
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
try:
    import umap.umap_ as umap
except ImportError:
    try:
        from umap import UMAP
        umap = type('umap', (), {'UMAP': UMAP})()
    except ImportError:
        umap = None
import matplotlib.pyplot as plt
import tarfile
import tempfile
import shutil
import re
from pathlib import Path


class VideoEmbeddingExtractor:
    """Extracts frame-level embeddings from videos using ResNet-50."""
    
    def __init__(self, device=None, skip_frames=4):
        """
        Initialize the embedding extractor.
        
        Args:
            device: torch device ('cuda' or 'cpu')
            skip_frames: Number of frames to skip between samples
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_frames = skip_frames
        
        # Load pre-trained ResNet-50 and remove classifier
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def extract_video_embedding(self, video_path):
        """
        Extract embedding for a single video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            numpy array: Mean embedding vector for the video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return None
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            # Skip frames
            for _ in range(self.skip_frames):
                ret = cap.read()[0]
                if not ret:
                    break
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            print(f"Warning: No frames extracted from {video_path}")
            return None
        
        # Extract embeddings for all frames
        embeddings = []
        with torch.no_grad():
            for frame in frames:
                try:
                    x = self.transform(frame).unsqueeze(0).to(self.device)
                    feat = self.model(x).squeeze().cpu().numpy()
                    # Flatten if needed
                    if feat.ndim > 1:
                        feat = feat.flatten()
                    embeddings.append(feat)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
        
        if len(embeddings) == 0:
            return None
        
        # Return mean embedding
        return np.mean(embeddings, axis=0)


def find_split_archives(data_dir, base_name):
    """
    Find all parts of a split archive.
    
    Args:
        data_dir: Directory to search
        base_name: Base name of the archive (e.g., "20bn-something-something-v2")
        
    Returns:
        list: Sorted list of archive part paths
    """
    archive_parts = []
    for file in os.listdir(data_dir):
        if file.startswith(base_name) and os.path.isfile(os.path.join(data_dir, file)):
            archive_parts.append(os.path.join(data_dir, file))
    
    # Sort to ensure correct order
    archive_parts.sort()
    return archive_parts


def concatenate_split_archives(archive_parts, output_path):
    """
    Concatenate split archive files into a single file.
    
    Args:
        archive_parts: List of archive part file paths
        output_path: Path to write concatenated file
        
    Returns:
        str: Path to concatenated file
    """
    print(f"Concatenating {len(archive_parts)} archive parts...")
    with open(output_path, 'wb') as outfile:
        for part_path in tqdm(archive_parts, desc="Concatenating"):
            with open(part_path, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
    print(f"Created concatenated archive: {output_path}")
    return output_path


def extract_videos_from_archive(archive_path, output_dir, max_videos=None):
    """
    Extract videos from tar archive. Handles various formats and split archives.
    
    Args:
        archive_path: Path to tar archive (or first part of split archive)
        output_dir: Directory to extract videos to
        max_videos: Maximum number of videos to extract (None for all)
        
    Returns:
        list: Paths to extracted video files
    """
    video_files = []
    video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
    
    print(f"Attempting to extract videos from {archive_path}...")
    
    # Try different tar formats
    tar_formats = ['r:*', 'r:gz', 'r:bz2', 'r:xz', 'r']
    
    for tar_format in tar_formats:
        try:
            with tarfile.open(archive_path, tar_format) as tar:
                members = tar.getmembers()
                video_members = [m for m in members if any(m.name.lower().endswith(ext) for ext in video_extensions)]
                
                if len(video_members) == 0:
                    # Try to list all members to see what's inside
                    print(f"Found {len(members)} files in archive, but no recognized video files.")
                    print("First 10 files in archive:")
                    for m in members[:10]:
                        print(f"  - {m.name}")
                    return []
                
                if max_videos:
                    video_members = video_members[:max_videos]
                
                print(f"Extracting {len(video_members)} video files...")
                for member in tqdm(video_members, desc="Extracting"):
                    try:
                        tar.extract(member, output_dir, filter='data')
                        extracted_path = os.path.join(output_dir, member.name)
                        if os.path.exists(extracted_path):
                            video_files.append(extracted_path)
                    except Exception as e:
                        print(f"Error extracting {member.name}: {e}")
                        continue
                
                print(f"Successfully extracted {len(video_files)} videos using format '{tar_format}'")
                return video_files
                
        except tarfile.TarError as e:
            continue
        except EOFError as e:
            # This might indicate a split archive
            print(f"EOFError with format '{tar_format}': {e}")
            continue
        except Exception as e:
            print(f"Error with format '{tar_format}': {e}")
            continue
    
    print(f"Could not extract from {archive_path} as a tar archive.")
    print("The file might be:")
    print("  1. A split archive (multi-part) that needs concatenation")
    print("  2. A raw video file or other format")
    print("  3. Corrupted or incomplete")
    return []


def find_video_files(directory):
    """
    Find all video files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        list: Paths to video files
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
    video_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    return video_files


def process_videos(video_files, extractor, output_file=None):
    """
    Process all videos and extract embeddings.
    
    Args:
        video_files: List of video file paths
        extractor: VideoEmbeddingExtractor instance
        output_file: Optional path to save embeddings
        
    Returns:
        tuple: (embeddings array, valid video files list)
    """
    video_embeddings = []
    valid_video_files = []
    
    print(f"Processing {len(video_files)} videos...")
    for video_path in tqdm(video_files, desc="Extracting embeddings"):
        embedding = extractor.extract_video_embedding(video_path)
        if embedding is not None:
            video_embeddings.append(embedding)
            valid_video_files.append(video_path)
    
    if len(video_embeddings) == 0:
        raise ValueError("No valid embeddings extracted!")
    
    video_embeddings = np.array(video_embeddings)
    print(f"Embedding shape: {video_embeddings.shape}")
    
    if output_file:
        np.savez(output_file, embeddings=video_embeddings, video_files=valid_video_files)
        print(f"Saved embeddings to {output_file}")
    
    return video_embeddings, valid_video_files


def cluster_videos(embeddings, n_clusters=10, use_pca=True, pca_components=128, 
                   normalize_embeddings=False, clustering_method='kmeans', **clustering_kwargs):
    """
    Cluster videos using various clustering methods.
    
    Args:
        embeddings: Video embeddings array
        n_clusters: Number of clusters (for K-Means, Agglomerative)
        use_pca: Whether to use PCA for dimensionality reduction
        pca_components: Number of PCA components
        normalize_embeddings: Whether to L2-normalize embeddings
        clustering_method: 'kmeans', 'dbscan', 'hdbscan', or 'agglomerative'
        **clustering_kwargs: Additional arguments for clustering method
        
    Returns:
        tuple: (cluster labels, reduced embeddings, PCA model, clustering model)
    """
    n_samples, n_features = embeddings.shape
    
    # Normalize embeddings if requested
    if normalize_embeddings:
        print("L2-normalizing embeddings...")
        embeddings = normalize(embeddings, axis=1, norm='l2')
    
    # Apply PCA if requested
    if use_pca and n_features > pca_components:
        # PCA can't have more components than samples
        # Use min(n_components, n_samples-1) to ensure it works
        max_components = min(pca_components, n_samples - 1, n_features)
        
        if max_components < pca_components:
            print(f"Warning: Requested {pca_components} PCA components, but only {n_samples} samples available.")
            print(f"Using {max_components} components instead.")
        
        if max_components > 0:
            print(f"Applying PCA: {n_features} -> {max_components} dimensions")
            pca = PCA(n_components=max_components, random_state=42)
            embeddings_reduced = pca.fit_transform(embeddings)
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        else:
            print("Skipping PCA: not enough samples for dimensionality reduction")
            embeddings_reduced = embeddings
            pca = None
    else:
        embeddings_reduced = embeddings
        pca = None
    
    # Apply clustering
    print(f"Clustering {len(embeddings_reduced)} videos using {clustering_method}...")
    
    if clustering_method == 'kmeans':
        # Adjust n_clusters if it's larger than number of samples
        actual_n_clusters = min(n_clusters, n_samples)
        if actual_n_clusters < n_clusters:
            print(f"Warning: Requested {n_clusters} clusters, but only {n_samples} samples available.")
            print(f"Using {actual_n_clusters} clusters instead.")
        
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10, **clustering_kwargs)
        labels = kmeans.fit_predict(embeddings_reduced)
        cluster_model = kmeans
        n_clusters = actual_n_clusters  # Update for return value
        
    elif clustering_method == 'dbscan':
        eps = clustering_kwargs.get('eps', 0.5)
        min_samples = clustering_kwargs.get('min_samples', 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **{k: v for k, v in clustering_kwargs.items() 
                                                             if k not in ['eps', 'min_samples']})
        labels = dbscan.fit_predict(embeddings_reduced)
        cluster_model = dbscan
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"DBSCAN found {n_clusters} clusters (+ {list(labels).count(-1)} noise points)")
        
    elif clustering_method == 'hdbscan':
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan not installed. Install with: pip install hdbscan")
        min_cluster_size = clustering_kwargs.get('min_cluster_size', 5)
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                        **{k: v for k, v in clustering_kwargs.items() 
                                           if k != 'min_cluster_size'})
        labels = hdbscan_model.fit_predict(embeddings_reduced)
        cluster_model = hdbscan_model
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"HDBSCAN found {n_clusters} clusters (+ {list(labels).count(-1)} noise points)")
        
    elif clustering_method == 'agglomerative':
        # Adjust n_clusters if it's larger than number of samples
        actual_n_clusters = min(n_clusters, n_samples)
        if actual_n_clusters < n_clusters:
            print(f"Warning: Requested {n_clusters} clusters, but only {n_samples} samples available.")
            print(f"Using {actual_n_clusters} clusters instead.")
        
        linkage = clustering_kwargs.get('linkage', 'ward')
        agglomerative = AgglomerativeClustering(n_clusters=actual_n_clusters, linkage=linkage,
                                                **{k: v for k, v in clustering_kwargs.items() 
                                                   if k != 'linkage'})
        labels = agglomerative.fit_predict(embeddings_reduced)
        cluster_model = agglomerative
        n_clusters = actual_n_clusters  # Update for return value
        
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    return labels, embeddings_reduced, pca, cluster_model


def build_faiss_index(embeddings, index_type='flat', use_cosine=False):
    """
    Build FAISS index for similarity search.
    
    Args:
        embeddings: Video embeddings array
        index_type: Type of FAISS index ('flat' or 'ivf')
        use_cosine: If True, use cosine similarity (inner product on normalized vectors)
        
    Returns:
        faiss.Index: FAISS index
    """
    d = embeddings.shape[1]
    # Ensure array is C-contiguous and float32
    embeddings = np.ascontiguousarray(embeddings, dtype='float32')
    
    # Normalize for cosine similarity
    if use_cosine:
        faiss.normalize_L2(embeddings)
    
    if index_type == 'flat' or len(embeddings) < 1000:
        # Use flat index for small datasets (exact search)
        if use_cosine:
            index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
        else:
            index = faiss.IndexFlatL2(d)  # L2 distance
    else:
        # Use IVF index for larger datasets (approximate, faster)
        nlist = min(100, len(embeddings) // 10)  # Number of clusters
        if use_cosine:
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
        else:
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(embeddings)
    
    index.add(embeddings)
    
    similarity_type = "cosine" if use_cosine else "L2"
    print(f"FAISS index built with {index.ntotal} vectors of dimension {d} ({similarity_type} similarity)")
    return index


def find_similar_videos(query_embedding, faiss_index, video_files, k=5):
    """
    Find k most similar videos to a query embedding.
    
    Args:
        query_embedding: Query video embedding (1D array)
        faiss_index: FAISS index
        video_files: List of video file paths
        k: Number of similar videos to return
        
    Returns:
        list: List of tuples (video_path, distance)
    """
    query = query_embedding.reshape(1, -1).astype('float32')
    k = min(k, len(video_files))
    
    D, I = faiss_index.search(query, k)
    
    results = [
        (video_files[idx], float(dist))
        for idx, dist in zip(I[0], D[0])
    ]
    
    return results


def visualize_clusters(embeddings, labels, output_file=None):
    """
    Visualize clusters using UMAP.
    
    Args:
        embeddings: Video embeddings (can be reduced)
        labels: Cluster labels
        output_file: Optional path to save plot
    """
    if umap is None:
        print("Warning: UMAP not available. Skipping visualization.")
        return
    
    print("Computing UMAP embedding for visualization...")
    # Adjust n_neighbors if we have fewer samples
    n_neighbors = min(15, len(embeddings) - 1) if len(embeddings) > 1 else 1
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        c=labels, 
        cmap='tab20', 
        s=30,
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title("Video Clusters Visualization (UMAP)", fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    
    plt.show()


def evaluate_cluster_quality(embeddings, labels):
    """
    Evaluate cluster quality using multiple metrics.
    
    Args:
        embeddings: Video embeddings array
        labels: Cluster labels
        
    Returns:
        dict: Dictionary of metric scores
    """
    print("\n" + "="*60)
    print("CLUSTER QUALITY EVALUATION")
    print("="*60)
    
    metrics = {}
    
    # Silhouette Score: Higher is better (range: -1 to 1)
    # Measures how similar videos are to their own cluster vs other clusters
    try:
        silhouette = silhouette_score(embeddings, labels)
        metrics['silhouette_score'] = silhouette
        print(f"\nSilhouette Score: {silhouette:.4f}")
        print("  (Range: -1 to 1, higher is better)")
        print("  Measures cohesion vs separation")
    except Exception as e:
        print(f"Could not compute Silhouette Score: {e}")
        metrics['silhouette_score'] = None
    
    # Calinski-Harabasz Index: Higher is better
    # Ratio of between-cluster to within-cluster variance
    try:
        calinski = calinski_harabasz_score(embeddings, labels)
        metrics['calinski_harabasz'] = calinski
        print(f"\nCalinski-Harabasz Index: {calinski:.2f}")
        print("  (Higher is better)")
        print("  Ratio of between-cluster to within-cluster variance")
    except Exception as e:
        print(f"Could not compute Calinski-Harabasz Index: {e}")
        metrics['calinski_harabasz'] = None
    
    # Davies-Bouldin Index: Lower is better
    # Average similarity ratio of clusters
    try:
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        metrics['davies_bouldin'] = davies_bouldin
        print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
        print("  (Lower is better)")
        print("  Average similarity ratio of clusters")
    except Exception as e:
        print(f"Could not compute Davies-Bouldin Index: {e}")
        metrics['davies_bouldin'] = None
    
    # Cluster size statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster Size Statistics:")
    print(f"  Number of clusters: {len(unique_labels)}")
    print(f"  Mean cluster size: {counts.mean():.1f}")
    print(f"  Std cluster size: {counts.std():.1f}")
    print(f"  Min cluster size: {counts.min()}")
    print(f"  Max cluster size: {counts.max()}")
    
    metrics['cluster_sizes'] = {
        'mean': float(counts.mean()),
        'std': float(counts.std()),
        'min': int(counts.min()),
        'max': int(counts.max())
    }
    
    return metrics


def find_cluster_representatives(embeddings, labels, video_files, n_representatives=3):
    """
    Find representative videos for each cluster (closest to cluster centroid).
    
    Args:
        embeddings: Video embeddings array
        labels: Cluster labels
        video_files: List of video file paths
        n_representatives: Number of representative videos per cluster
        
    Returns:
        dict: Dictionary mapping cluster_id to list of representative video paths
    """
    representatives = {}
    unique_labels = np.unique(labels)
    
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_video_indices = np.where(cluster_mask)[0]
        
        # Compute cluster centroid
        centroid = cluster_embeddings.mean(axis=0)
        
        # Find videos closest to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_indices = np.argsort(distances)[:n_representatives]
        
        representatives[cluster_id] = [
            (video_files[cluster_video_indices[idx]], float(distances[idx]))
            for idx in closest_indices
        ]
    
    return representatives


def print_cluster_summary(labels, video_files, n_clusters=None, representatives=None):
    """Print summary of clusters."""
    print("\n" + "="*60)
    print("CLUSTER SUMMARY")
    print("="*60)
    
    unique_labels = np.unique(labels)
    # Filter out noise label (-1) for DBSCAN/HDBSCAN
    cluster_ids = [l for l in unique_labels if l != -1]
    noise_count = list(labels).count(-1) if -1 in labels else 0
    
    if noise_count > 0:
        print(f"\nNoise points (unclustered): {noise_count}")
    
    # Use actual number of clusters if not specified
    if n_clusters is None:
        n_clusters = len(cluster_ids)
    
    for cluster_id in sorted(cluster_ids):
        cluster_videos = [video_files[i] for i, label in enumerate(labels) if label == cluster_id]
        print(f"\nCluster {cluster_id}: {len(cluster_videos)} videos")
        
        # Print representatives if available
        if representatives and cluster_id in representatives:
            print("  Representatives (closest to centroid):")
            for vid_path, dist in representatives[cluster_id]:
                print(f"    - {os.path.basename(vid_path)} (distance: {dist:.2f})")
        
        if len(cluster_videos) <= 10:
            for vid in cluster_videos:
                print(f"  - {os.path.basename(vid)}")
        else:
            for vid in cluster_videos[:5]:
                print(f"  - {os.path.basename(vid)}")
            print(f"  ... and {len(cluster_videos) - 5} more")


def main():
    """Main pipeline execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Clustering Pipeline")
    parser.add_argument("--data-dir", type=str, default=".", 
                       help="Directory containing video files or archives")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory for results")
    parser.add_argument("--n-clusters", type=int, default=10,
                       help="Number of clusters")
    parser.add_argument("--max-videos", type=int, default=None,
                       help="Maximum number of videos to process")
    parser.add_argument("--skip-frames", type=int, default=4,
                       help="Frames to skip between samples")
    parser.add_argument("--pca-components", type=int, default=128,
                       help="Number of PCA components")
    parser.add_argument("--no-pca", action="store_true",
                       help="Disable PCA dimensionality reduction")
    parser.add_argument("--extract-archives", action="store_true",
                       help="Extract videos from tar archives")
    parser.add_argument("--load-embeddings", type=str, default=None,
                       help="Load pre-computed embeddings from file")
    parser.add_argument("--normalize", action="store_true",
                       help="L2-normalize embeddings before clustering")
    parser.add_argument("--clustering-method", type=str, default="kmeans",
                       choices=['kmeans', 'dbscan', 'hdbscan', 'agglomerative'],
                       help="Clustering method to use")
    parser.add_argument("--dbscan-eps", type=float, default=0.5,
                       help="DBSCAN eps parameter")
    parser.add_argument("--dbscan-min-samples", type=int, default=5,
                       help="DBSCAN min_samples parameter")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=5,
                       help="HDBSCAN min_cluster_size parameter")
    parser.add_argument("--use-cosine", action="store_true",
                       help="Use cosine similarity for FAISS index (requires --normalize)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_cosine and not args.normalize:
        print("Warning: --use-cosine requires --normalize. Enabling normalization.")
        args.normalize = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = VideoEmbeddingExtractor(skip_frames=args.skip_frames)
    print(f"Using device: {extractor.device}")
    
    # Find or extract videos
    if args.load_embeddings and os.path.exists(args.load_embeddings):
        print(f"Loading embeddings from {args.load_embeddings}")
        data = np.load(args.load_embeddings, allow_pickle=True)
        video_embeddings = data['embeddings']
        video_files = data['video_files'].tolist()
    else:
        video_files = []
        
        # First, try to find video files directly in directory
        video_files = find_video_files(args.data_dir)
        
        # If no videos found, check for archive files
        if len(video_files) == 0:
            archive_files = [
                f for f in os.listdir(args.data_dir) 
                if f.startswith("20bn-something-something") and os.path.isfile(os.path.join(args.data_dir, f))
            ]
            
            if archive_files and args.extract_archives:
                # Check if these are split archives
                # Group files by base name (e.g., "20bn-something-something-v2")
                base_names = set()
                for archive_file in archive_files:
                    # Extract base name (remove trailing numbers like -00, -01)
                    # Files like "20bn-something-something-v2-00" should become "20bn-something-something-v2"
                    file_no_ext = archive_file.rsplit('.', 1)[0] if '.' in archive_file else archive_file
                    # Check if it ends with a pattern like -00, -01 (two digits)
                    match = re.match(r'^(.+)-(\d{2,})$', file_no_ext)
                    if match:
                        base_name = match.group(1)
                    else:
                        base_name = file_no_ext
                    base_names.add(base_name)
                
                temp_dir = tempfile.mkdtemp()
                concatenated_archives = []
                
                try:
                    for base_name in base_names:
                        # Find all parts of this archive
                        archive_parts = find_split_archives(args.data_dir, base_name)
                        
                        if len(archive_parts) > 1:
                            # This is a split archive - concatenate first
                            print(f"\nDetected split archive with {len(archive_parts)} parts")
                            print("Parts:", [os.path.basename(p) for p in archive_parts])
                            
                            concat_path = os.path.join(temp_dir, f"{base_name}_concatenated.tar")
                            if not os.path.exists(concat_path):
                                concatenate_split_archives(archive_parts, concat_path)
                            concatenated_archives.append(concat_path)
                        else:
                            # Single archive file
                            concatenated_archives.extend(archive_parts)
                    
                    # Extract from concatenated or single archives
                    for archive_path in concatenated_archives:
                        extracted = extract_videos_from_archive(
                            archive_path, temp_dir, args.max_videos
                        )
                        video_files.extend(extracted)
                        if args.max_videos and len(video_files) >= args.max_videos:
                            video_files = video_files[:args.max_videos]
                            break
                            
                except Exception as e:
                    print(f"Error during archive extraction: {e}")
                    import traceback
                    traceback.print_exc()
                    print("\nTrying alternative: checking if files are raw video data...")
                    # Fallback: try to read as video files
                    for archive_file in archive_files:
                        archive_path = os.path.join(args.data_dir, archive_file)
                        cap = cv2.VideoCapture(archive_path)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            cap.release()
                            if ret:
                                print(f"File {archive_file} appears to be a video file!")
                                video_files.append(archive_path)
                finally:
                    # Optionally clean up concatenated files (they're large, so maybe keep them)
                    # shutil.rmtree(temp_dir, ignore_errors=True)
                    pass
            elif archive_files:
                # Try to treat archive files as video files (they might be video files, not archives)
                print("Found files that might be archives. Trying to process as video files...")
                print("(Use --extract-archives if these are tar archives)")
                video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
                for archive_file in archive_files:
                    archive_path = os.path.join(args.data_dir, archive_file)
                    # Check if file extension suggests it's a video
                    if any(archive_file.lower().endswith(ext) for ext in video_extensions):
                        video_files.append(archive_path)
                    # Try to open as video to verify
                    else:
                        cap = cv2.VideoCapture(archive_path)
                        if cap.isOpened():
                            video_files.append(archive_path)
                            cap.release()
        
        if args.max_videos:
            video_files = video_files[:args.max_videos]
        
        if len(video_files) == 0:
            print("No video files found!")
            return
        
        print(f"Found {len(video_files)} video files")
        
        # Extract embeddings
        embedding_file = os.path.join(args.output_dir, "video_embeddings.npz")
        video_embeddings, video_files = process_videos(
            video_files, extractor, embedding_file
        )
    
    # Cluster videos
    clustering_kwargs = {}
    if args.clustering_method == 'dbscan':
        clustering_kwargs['eps'] = args.dbscan_eps
        clustering_kwargs['min_samples'] = args.dbscan_min_samples
    elif args.clustering_method == 'hdbscan':
        clustering_kwargs['min_cluster_size'] = args.hdbscan_min_cluster_size
    
    labels, embeddings_reduced, pca, cluster_model = cluster_videos(
        video_embeddings,
        n_clusters=args.n_clusters,
        use_pca=not args.no_pca,
        pca_components=args.pca_components,
        normalize_embeddings=args.normalize,
        clustering_method=args.clustering_method,
        **clustering_kwargs
    )
    
    # Evaluate cluster quality
    metrics = evaluate_cluster_quality(embeddings_reduced, labels)
    
    # Find cluster representatives
    print("\nFinding cluster representatives...")
    representatives = find_cluster_representatives(embeddings_reduced, labels, video_files, n_representatives=3)
    
    # Print cluster summary with representatives
    # Get actual number of clusters (may differ for DBSCAN/HDBSCAN)
    n_actual_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_actual_clusters > 0:  # Only print if we have clusters
        print_cluster_summary(labels, video_files, n_actual_clusters, representatives)
    else:
        print("\nNote: All videos are in a single cluster or unclustered.")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    faiss_index = build_faiss_index(embeddings_reduced, use_cosine=args.use_cosine)
    
    # Example: Find nearest neighbors
    if len(embeddings_reduced) > 0:
        print("\n" + "="*60)
        print("SIMILARITY SEARCH EXAMPLE")
        print("="*60)
        print("\nFinding 5 nearest neighbors for first video...")
        similar_videos = find_similar_videos(
            embeddings_reduced[0], 
            faiss_index, 
            video_files, 
            k=5
        )
        print(f"Query video: {os.path.basename(video_files[0])}")
        print("\nMost similar videos:")
        for vid_path, dist in similar_videos:
            print(f"  - {os.path.basename(vid_path)} (distance: {dist:.2f})")
    
    # Visualize
    viz_file = os.path.join(args.output_dir, "cluster_visualization.png")
    visualize_clusters(embeddings_reduced, labels, viz_file)
    
    # Save results
    results_file = os.path.join(args.output_dir, "clustering_results.npz")
    
    # Save PCA model if it exists
    import pickle
    pca_file = os.path.join(args.output_dir, "pca_model.pkl")
    if pca is not None:
        with open(pca_file, 'wb') as f:
            pickle.dump(pca, f)
        print(f"Saved PCA model to {pca_file}")
    
    np.savez(
        results_file,
        labels=labels,
        embeddings=embeddings_reduced,
        video_files=video_files,
        metrics=metrics,
        original_embeddings_shape=video_embeddings.shape if 'video_embeddings' in locals() else None,
        normalize=args.normalize,
        use_cosine=args.use_cosine
    )
    print(f"\nSaved clustering results to {results_file}")
    
    # Save FAISS index
    faiss_index_file = os.path.join(args.output_dir, "faiss_index.bin")
    faiss.write_index(faiss_index, faiss_index_file)
    print(f"Saved FAISS index to {faiss_index_file}")


if __name__ == "__main__":
    main()


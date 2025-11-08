"""
Standalone script to evaluate cluster quality from saved results.
"""

import os
import numpy as np
import argparse
from video_clustering import evaluate_cluster_quality, print_cluster_summary, find_cluster_representatives


def main():
    parser = argparse.ArgumentParser(description="Evaluate cluster quality")
    parser.add_argument("--results-file", type=str, default="output/clustering_results.npz",
                       help="Path to clustering results file")
    parser.add_argument("--n-representatives", type=int, default=3,
                       help="Number of representative videos per cluster")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    print(f"Loading clustering results from {args.results_file}...")
    data = np.load(args.results_file, allow_pickle=True)
    
    embeddings = data['embeddings']
    labels = data['labels']
    video_files = data['video_files'].tolist()
    
    # Get number of clusters
    n_clusters = len(np.unique(labels))
    
    print(f"Loaded {len(embeddings)} videos with {n_clusters} clusters")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Evaluate cluster quality
    metrics = evaluate_cluster_quality(embeddings, labels)
    
    # Find cluster representatives
    print("\nFinding cluster representatives...")
    representatives = find_cluster_representatives(
        embeddings, labels, video_files, 
        n_representatives=args.n_representatives
    )
    
    # Print cluster summary
    print_cluster_summary(labels, video_files, n_clusters, representatives)
    
    # Print metrics summary
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    if metrics.get('silhouette_score') is not None:
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    if metrics.get('calinski_harabasz') is not None:
        print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.2f}")
    if metrics.get('davies_bouldin') is not None:
        print(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")


if __name__ == "__main__":
    main()


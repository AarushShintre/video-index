"""
Quick start script for video clustering.
This is a simplified version for quick testing.
"""

import subprocess
import sys
import os

def main():
    print("="*60)
    print("Video Clustering - Quick Start")
    print("="*60)
    print()
    
    # Check if required packages are installed
    print("Checking dependencies...")
    try:
        import torch
        import cv2
        import sklearn
        import faiss
        import umap
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return
    
    print()
    print("Starting video clustering pipeline...")
    print("Processing videos in current directory...")
    print()
    
    # Run with reasonable defaults for quick start
    cmd = [
        sys.executable,
        "video_clustering.py",
        "--max-videos", "50",  # Start with 50 videos for quick testing
        "--n-clusters", "5",
        "--output-dir", "output"
    ]
    
    # Check if archive files exist and suggest extraction
    archive_files = [
        f for f in os.listdir(".") 
        if f.startswith("20bn-something-something") and os.path.isfile(f)
    ]
    
    if archive_files:
        print(f"Found {len(archive_files)} potential archive file(s)")
        response = input("Do you want to extract videos from archives? (y/n): ")
        if response.lower() == 'y':
            cmd.append("--extract-archives")
    
    print()
    print("Running command:", " ".join(cmd))
    print()
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()


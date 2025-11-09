import os
import sys
import json
import cv2
import torch
import numpy as np
import faiss
from torchvision import models, transforms
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_embedding(video_path, model, preprocess):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while count < 16:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess(frame)
        frames.append(frame)
        count += 1

    cap.release()

    if len(frames) == 0:
        return None

    batch = torch.stack(frames).to(device)
    with torch.no_grad():
        embedding = model(batch).cpu().numpy()

    return embedding.mean(axis=0)

def load_models(results_dir):
    global model, preprocess, faiss_index, pca_model, video_files, normalize_embeddings, use_cosine

    # Load ResNet
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()
    model = resnet

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    embeddings_dir = os.path.join(results_dir, "embeddings")
    index_path = os.path.join(embeddings_dir, "faiss_index.bin")
    cluster_path = os.path.join(embeddings_dir, "clustering_results.npz")
    pca_path = os.path.join(embeddings_dir, "pca_model.pkl")

    # Load FAISS index
    if os.path.exists(index_path):
        faiss_index = faiss.read_index(index_path)
    else:
        faiss_index = None

    # Load PCA if exists
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca_model = pickle.load(f)
    else:
        pca_model = None

    # Load metadata
    if os.path.exists(cluster_path):
        data = np.load(cluster_path, allow_pickle=True)
        video_files = list(data["video_files"])
        normalize_embeddings = bool(data["normalize"])
        use_cosine = bool(data["use_cosine"])
    else:
        video_files = []
        normalize_embeddings = False
        use_cosine = False

def save_updated_index(results_dir):
    embeddings_dir = os.path.join(results_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    # Save FAISS
    faiss.write_index(faiss_index, os.path.join(embeddings_dir, "faiss_index.bin"))

    # Save metadata
    np.savez(
        os.path.join(embeddings_dir, "clustering_results.npz"),
        video_files=np.array(video_files, dtype=object),
        normalize=normalize_embeddings,
        use_cosine=use_cosine
    )

def main():
    args = json.loads(sys.argv[1])
    operation = args.get("operation")
    results_dir = args.get("results_dir")
    video_path = args.get("video_path")

    load_models(results_dir)

    # ✅ Add new video to FAISS index
    if operation == "add_to_index":
        if not os.path.exists(video_path):
            print(json.dumps({"error": "video not found"}))
            return
        
        embedding = extract_embedding(video_path, model, preprocess)
        if embedding is None:
            print(json.dumps({"error": "could not extract embedding"}))
            return

        emb = embedding.astype("float32")

        # PCA if needed
        if "pca_model" in globals() and pca_model is not None:
            emb = pca_model.transform(emb.reshape(1, -1)).flatten()

        # Normalize if enabled
        if normalize_embeddings:
            from sklearn.preprocessing import normalize
            emb = normalize(emb.reshape(1, -1), norm="l2").flatten()

        # Convert to FAISS shape
        vec = emb.reshape(1, -1).astype("float32")

        # Create index if first entry
        if faiss_index is None:
            dims = vec.shape[1]
            faiss_index = faiss.IndexFlatL2(dims)

        faiss_index.add(vec)
        video_files.append(video_path)

        save_updated_index(results_dir)

        print(json.dumps({"success": True, "index_size": faiss_index.ntotal}))

    # ✅ Existing search operation (unchanged)
    elif operation == "search":
        query = args.get("query_vector")
        vec = np.array(query, dtype="float32").reshape(1, -1)
        D, I = faiss_index.search(vec, 5)

        results = []
        for idx in I[0]:
            if idx < len(video_files):
                results.append(video_files[idx])

        print(json.dumps({"results": results}))

if __name__ == "__main__":
    main()

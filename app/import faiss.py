import faiss
import json
import numpy as np
import os


def build_faiss_index(
    chunks, index_path="data/faiss.index", meta_path="data/metadata.json"
):
    """
    Builds and saves FAISS index with cosine similarity.
    """

    embeddings = []
    metadata = []

    for i, chunk in enumerate(chunks):
        embeddings.append(chunk["embedding"])
        metadata.append(
            {
                "text": chunk["chunk_text"],
                "page": chunk.get("metadata", {}).get("page"),
                "source": chunk.get("metadata", {}).get("source"),
            }
        )

    embeddings = np.array(embeddings).astype("float32")

    # 🔥 Normalize embeddings (IMPORTANT)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # 🔥 Use Inner Product (cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Ensure folder exists
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Save index
    faiss.write_index(index, index_path)

    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ FAISS index built and saved.")

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Global cache (avoid reloading every time)
index = None
metadata = None


def load_index(index_path="data/faiss.index", meta_path="data/metadata.json"):
    global index, metadata

    if index is None:
        index = faiss.read_index(index_path)

    if metadata is None:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return index, metadata


def retrieve(query, top_k=5):
    """
    Retrieves top_k most relevant chunks using cosine similarity.
    """

    index, metadata = load_index()

    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)
    if isinstance(
        query_embedding, np.ndarray
    ):  # Ensure the embedding is a numpy array and if it is, then
        query_embedding = query_embedding.astype("float32")
    else:  # Convert to numpy array if not already
        query_embedding = np.array(query_embedding, dtype="float32")

    # 🔥 Normalize query (CRITICAL)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        score = float(distances[0][rank])

        # Optional filter (remove weak matches)
        if score < 0.3:
            continue

        results.append(
            {
                "chunk_text": metadata[idx]["text"],
                "chunk_id": idx,
                "score": score,  # higher = better
                "metadata": {
                    "page": metadata[idx].get("page"),
                    "source": metadata[idx].get("source"),
                },
            }
        )

    return results

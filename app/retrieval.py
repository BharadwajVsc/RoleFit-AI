import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_index(index_path="data/faiss.index", meta_path="data/metadata.json"):
    """Loads a FAISS index and its metadata from disk.

    Args:
        index_path (str): The file path of the FAISS index.
        meta_path (str): The file path of the metadata.

    Returns:
        tuple: A tuple containing the FAISS index and metadata list.
    """
    index = faiss.read_index(index_path)  # Load the FAISS index from disk
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)  # Load metadata from disk

    return index, metadata


def retrieve(query, top_k=5):
    """Retrieves the top_k most similar text chunks for a given query.

    Args:
        query (str): The input query string.
        top_k (int): The number of top similar chunks to retrieve.
    Returns:
        list: A list of dictionaries containing retrieved chunks,
        similarity scores, and metadata.
    """

    index, metadata = load_index()  # Load the FAISS index and metadata
    query_embedding = model.encode([query])  # Generate embedding for the query
    if isinstance(
        query_embedding, np.ndarray
    ):  # Ensure the embedding is a numpy array and if it is, then
        query_embedding = query_embedding.astype("float32")
    else:  # Convert to numpy array if not already
        query_embedding = np.array(query_embedding, dtype="float32")

    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(
        query_embedding, top_k
    )  # Search the index for similar chunks
    results = []

    for rank, idx in enumerate(indices[0]):  # Iterate over the retrieved indices
        if idx == -1:  # ``
            continue
        results.append(
            {
                "chunk_text": metadata[idx]["text"],
                "chunk_id": idx,
                "score": float(distances[0][rank]),
                "metadata": {
                    "page": metadata[idx].get("page", None),
                    "source": metadata[idx].get("source", None),
                },
            }
        )  # Append chunk details to results

    return results

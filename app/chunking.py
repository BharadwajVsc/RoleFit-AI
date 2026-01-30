def chunking(text: str, chunk_size: int = 700, overlap: int = 100) -> list:
    """Splits text into overlapping chunks.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        list: A list of text chunks.
    """

    if not text:
        return []
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size  # Define end of chunk
        chunk_text = text[start:end]  # Extract chunk text
        chunks.append(
            {"chunk_id": chunk_id, "text": chunk_text}
        )  # Append chunk to list

        chunk_id = chunk_id + 1  # Increment chunk ID
        start = end - overlap  # Move start forward with overlap
    return chunks

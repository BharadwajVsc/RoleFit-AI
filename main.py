from app.ingestion import upload_pdf
from app.chunking import chunking
from app.embeddings import generate_embeddings
from app.vector_storage import build_faiss_index
from app.retrieval import retrieve
from app.llm_reasoning import llm_reasoning
from app.llm_client import LLMClient

doc = upload_pdf(r"D:\interview prep\Bharadwaj VSC Resume.pdf")
chunks = chunking(doc["extracted_text"])
print(f"Total Chunks: {len(chunks)}")  # this will print the total number of chunks

chunks = generate_embeddings(chunks)  # this will generate embeddings for all the chunks

build_faiss_index(chunks)  # this will build the faiss index and save it to disk

query = "What skills are required for generative ai role?"
retrieved_chunks = retrieve(
    query, top_k=3
)  # this will retrieve top 3 relevant chunks for the query

print("\nRetrieved Chunks:\n")
for i, res in enumerate(retrieved_chunks, 1):  # this will print the retrieved chunks
    print(f"{i}.{res}\n")  # res is the retrieved chunk text

print("\n---Retrieval Debug ---")  # this will print debug info for retrieval
for i, item in enumerate(
    retrieved_chunks, start=1
):  # this will print each retrieved chunk with its score and metadata
    preview = item["chunk_text"][:120].replace(
        "\n", " "
    )  # this will create a preview of the chunk text
    print(
        f"Rank {i} | score: {item['score']:.4f} |", f"page: {item['metadata']['page']}"
    )  # this will print rank, score and page number
    print(f"Preview: {preview}...\n")
print("---------------------------\n")

llm_client = LLMClient()
final_response = llm_reasoning(
    retrieved_chunks=retrieved_chunks, query=query, llm_client=llm_client
)
print('\nFinal LLM Response"\n')
print(final_response)  # this will print the final response from the LLM

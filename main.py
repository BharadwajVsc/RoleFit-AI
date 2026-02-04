from app.ingestion import upload_pdf, ingest_jd
from app.chunking import chunking
from app.embeddings import generate_embeddings
from app.vector_storage import build_faiss_index
from app.retrieval import retrieve
from app.llm_reasoning import llm_reasoning, extract_jd_requirements
from app.llm_client import LLMClient
from app.matcher import match_skills, matched_responsibilities, evaluate_match_score
import json


def clean_llm_json(text: str) -> str:
    """
    Extract the first valid JSON object from LLM output.
    """
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output")

    return text[start : end + 1]


llm_client = LLMClient()

doc = upload_pdf(r"C:\Users\bhara\Downloads\Rohan_Chinta.pdf")
jd_text = "We are looking for a skilled Generative AI Engineer to join our innovative team. The ideal candidate will have experience in developing and deploying generative AI models, with a strong background in machine learning, deep learning, and natural language processing. Responsibilities include designing AI architectures, training models on large datasets, and collaborating with cross-functional teams to integrate AI solutions into products. Proficiency in Python, TensorFlow, PyTorch, and experience with cloud platforms such as AWS or GCP is required. The candidate should also have excellent problem-solving skills and the ability to stay updated with the latest advancements in AI technology."
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


final_response = llm_reasoning(
    retrieved_chunks=retrieved_chunks, query=query, llm_client=llm_client
)
print('\nFinal LLM Response"\n')
print(final_response)  # this will print the final response from the LLM

jd = ingest_jd(jd_text)  # this will ingest the job description text
structured_jd = extract_jd_requirements(
    jd, llm_client
)  # this will extract structured requirements from the JD using LLM

print("\nRAW LLM OUTPUT:")
print(structured_jd)

structured_jd = clean_llm_json(structured_jd)
structured_jd = json.loads(structured_jd)  # parse the JSON string to dict
print("\n ---- Structured JD Requirements ---- \n")
print(structured_jd)  # this will print the structured JD requirements

required_skills = structured_jd["required_skills"]
responsibilities = structured_jd["responsibilities"]

matched_skills, missing_skills, skill_evidence = match_skills(required_skills, top_k=3)

matched_resps, resp_evidence = matched_responsibilities(responsibilities, top_k=3)

final_score = evaluate_match_score(
    matched_skills, len(required_skills), matched_resps, len(responsibilities)
)

print("\n---- Skill Matching Results ----\n")
print(f"Matched Sore:{final_score}%\n")

print("\nMatched Skills:")
for s in matched_skills:
    print(f"- {s}")

print("\nMissing Skills:")
for s in missing_skills:
    print(f"- {s}")

print("\nMatched Responsibilities:")
for r in matched_resps:
    print(f"- {r}")

from app.ingestion import upload_pdf, ingest_jd
from app.chunking import chunking
from app.embeddings import generate_embeddings
from app.vector_storage import build_faiss_index
from app.retrieval import retrieve
from app.llm_reasoning import llm_reasoning, extract_jd_requirements
from app.llm_client import LLMClient
from app.matcher import match_skills, matched_responsibilities, evaluate_match_score
from app.explanation import generate_explanation
import json


import re


def clean_llm_json(text: str) -> str:
    """
    Extract and fix JSON from LLM output safely.
    """

    # Remove markdown ```json ```
    text = re.sub(r"```json|```", "", text).strip()

    # Extract JSON block
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output")

    json_text = text[start : end + 1]

    # Fix common JSON issues
    json_text = re.sub(r",\s*}", "}", json_text)  # remove trailing comma
    json_text = re.sub(r",\s*]", "]", json_text)  # remove trailing comma

    return json_text


llm_client = LLMClient()

doc = upload_pdf(r"D:\interview\Bharadwaj VSC Resume.pdf")
jd_text = "Software Engineer specializing in Generative AI and AI-powered platforms with 3+ years of enterprise experience and hands-on expertise in building production-grade LLM applications. Experienced in developing scalable AI systems using LangChain, Retrieval-Augmented Generation (RAG), and modern LLMs such as Google Gemini and OpenAI. Skilled in designing prompt-driven workflows, building FastAPI-based AI services, and creating NLP pipelines for automation, data extraction, and intelligent decision support. Strong background in Python, backend development, and database systems, with experience delivering AI-driven solutions for HR Tech and FinTech domains. Proven ability to optimize AI pipelines for performance, reliability, and accuracy while building modular, production-ready architectures."
chunks = chunking(doc["extracted_text"])
print(f"Total Chunks: {len(chunks)}")  # this will print the total number of chunks
print(chunks)

chunks = generate_embeddings(chunks)  # this will generate embeddings for all the chunks

build_faiss_index(chunks)  # this will build the faiss index and save it to disk

query = "Extract and rank the key skills required for this role. Separate them into must-have and good-to-have skills."
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

jd = ingest_jd(jd_text)  # this will ingest the job description text
structured_jd = extract_jd_requirements(
    jd, llm_client
)  # this will extract structured requirements from the JD using LLM

print("\nRAW LLM OUTPUT:")
print(structured_jd)

structured_jd = clean_llm_json(structured_jd)
# print("\nCleaned JSON String:")
# print(structured_jd)  # this will print the cleaned JSON string
structured_jd = json.loads(structured_jd)  # parse the JSON string to dict
print("\n ---- Structured JD Requirements ---- \n")
print(structured_jd)  # this will print the structured JD requirements

print("\n--- Calling LLM ---\n")

final_response = llm_reasoning(
    retrieved_chunks=retrieved_chunks,
    query=query,
    llm_client=llm_client,
    structured_jd=structured_jd,
)
print('\nFinal LLM Response"\n')
print(final_response)  # this will print the final response from the LLM
print('\nFinal LLM Response ENDDDDDDDDDDDDDDDDD"\n')


required_skills = structured_jd["required_skills"]
responsibilities = structured_jd["responsibilities"]

matched_skills, missing_skills, skill_evidence = match_skills(required_skills, top_k=8)

matched_resps, resp_evidence = matched_responsibilities(responsibilities, top_k=3)

final_score = evaluate_match_score(
    matched_skills, len(required_skills), matched_resps, len(responsibilities)
)

explanation = generate_explanation(
    final_score, matched_skills, missing_skills, matched_resps
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

results = {
    "match_score": final_score,
    "matched_skills": matched_skills,
    "missing_skills": missing_skills,
    "matched_responsibilities": matched_resps,
    "strengths": explanation["strengths"],
    "weaknesses": explanation["weaknesses"],
    "recommendation": explanation["recommendation"],
    "improvement_suggestions": explanation["improvement_suggestions"],
}
print("\n---- Final Evaluation ----\n")
print(json.dumps(results, indent=2))

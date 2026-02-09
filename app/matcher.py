from app.retrieval import retrieve

# Slightly stricter threshold to avoid weak semantic matches
SKILL_SCORE_THRESHOLD = 1.12
RESP_SCORE_THRESHOLD = 1.18


# ---------- Normalization ----------
def normalize(text: str) -> str:
    text = text.lower()

    replacements = {
        "ml": "machine learning",
        "nlp": "natural language processing",
        "genai": "generative ai",
        "llm": "generative ai",
        "rag": "retrieval augmented generation",
        "tensorflow keras": "tensorflow",
        "pytorch lightning": "pytorch",
        "gcp": "google cloud",
        "aws": "amazon web services",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace("models", "")
    text = text.replace("model", "")

    return " ".join(text.split())


# ---------- Skill Matching ----------
def match_skills(required_skills: list, top_k: int = 3):

    AI_CORE = {
        "generative ai",
        "natural language processing",
        "deep learning",
        "machine learning",
    }

    resume_text = " ".join([r["chunk_text"].lower() for r in retrieve("ai", top_k=8)])
    ai_presence = any(skill in resume_text for skill in AI_CORE)

    if not ai_presence:
        return [], required_skills, {}

    matched = []
    missing = []
    evidence = {}

    for skill in required_skills:

        results = retrieve(skill, top_k=top_k)

        # Weak retrieval → missing
        if not results or results[0]["score"] > 1.35:
            missing.append(skill)
            continue

        norm_skill = normalize(skill)
        keyword_found = False

        # ---------- Keyword grounding ----------
        for r in results:
            chunk_text = normalize(r["chunk_text"])

            if norm_skill in chunk_text and results[0]["score"] <= 1.28:
                matched.append(skill)
                evidence[skill] = r["chunk_text"]
                keyword_found = True
                break

        if keyword_found:
            continue

        # ---------- Strong semantic fallback ----------
        if results[0]["score"] <= 1.30:
            matched.append(skill)
            evidence[skill] = results[0]["chunk_text"]
        else:
            missing.append(skill)

    # Remove duplicates (safety)
    matched = list(dict.fromkeys(matched))
    missing = list(dict.fromkeys(missing))

    return matched, missing, evidence


# ---------- Responsibility Matching ----------
def matched_responsibilities(responsibilities: list, top_k: int = 3):
    matched = []
    evidence = {}

    for resp in responsibilities:
        results = retrieve(resp, top_k=top_k)

        if results and results[0]["score"] <= RESP_SCORE_THRESHOLD:
            matched.append(resp)
            evidence[resp] = results[0]["chunk_text"]

    return matched, evidence


# ---------- Final Score ----------
def evaluate_match_score(
    matched_skills, total_skills, matched_responsibilities, total_responsibilities
):
    skill_score = len(matched_skills) / total_skills if total_skills > 0 else 0

    resp_score = (
        len(matched_responsibilities) / total_responsibilities
        if total_responsibilities > 0
        else 0
    )

    final_score = (0.8 * skill_score + 0.2 * resp_score) * 100
    GENAI_CORE = {
        "generative ai",
        "natural language processing",
        "retrieval augmented generation",
        "langchain",
    }

    bonus = 0
    for skill in matched_skills:
        if normalize(skill) in GENAI_CORE:
            bonus += 2  # +2% each

    final_score = min(final_score + bonus, 100)
    return round(final_score, 2)

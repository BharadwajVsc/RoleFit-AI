from app.retrieval import retrieve

SKILL_SCORE_THRESHOLD = 1.2  # skill score threshold for reasoning
RESP_SCORE_THRESHOLD = 1.3  # response score threshold for reasoning


def match_skills(required_skills: list, top_k: int = 3):

    matched = []
    missing = []
    evidence = {}

    for skill in required_skills:
        results = retrieve(skill, top_k=top_k)
        if not results or results[0]["score"] < SKILL_SCORE_THRESHOLD:
            missing.append(skill)
        else:
            matched.append(skill)
            evidence[skill] = results[0]["chunk_text"]

    return matched, missing, evidence


def matched_responsibilities(responsibilities: list, top_k: int = 3):
    matched = []
    evidence = {}

    for resp in responsibilities:
        results = retrieve(resp, top_k=top_k)

        if results and results[0]["score"] >= RESP_SCORE_THRESHOLD:
            matched.append(resp)
            evidence[resp] = results[0]["chunk_text"]

    return matched, evidence


def evaluate_match_score(
    matched_skills, total_skills, matched_responsibilities, total_responsibilities
):
    skill_score = len(matched_skills) / total_skills if total_skills > 0 else 0

    resp_score = (
        len(matched_responsibilities) / total_responsibilities
        if total_responsibilities > 0
        else 0
    )

    final_score = (0.7 * skill_score + 0.3 * resp_score) * 100
    return round(final_score, 2)

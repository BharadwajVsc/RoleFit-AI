def generate_explanation(
    matched_score: float,
    matched_skills: list,
    missing_skills: list,
    matched_responsibilities: list,
):
    """
    Will generate an explanation based on the matched skills, missing skills, and matched responsibilities.
    """

    strengths = []
    if matched_skills:
        strengths.append(
            f"Has strong technical match in the following skills:{','.join(matched_skills[:5])}"
        )
    if matched_responsibilities:
        strengths.append(f"Demonstrated experience with key responsibilities")

    weaknesses = []
    if missing_skills:
        weaknesses.append(
            f"Missing the following important skills: {','.join(missing_skills[:5])}"
        )
    else:
        weaknesses.append("No significant skill gaps identified.")

    if matched_score >= 80:
        recommendation = "Excellent fit for the role."
    elif matched_score >= 60:
        recommendation = "Good fit, but there are some areas for improvement."
    else:
        recommendation = (
            "Not a strong fit for the role based on current skills and experience."
        )

    if missing_skills:
        improvement_suggestions = f"To improve fit, consider gaining experience in: {','.join(missing_skills[:5])}."
    else:
        improvement_suggestions = "No specific improvement suggestions, candidate is well-aligned with the role."

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendation": recommendation,
        "improvement_suggestions": improvement_suggestions,
    }

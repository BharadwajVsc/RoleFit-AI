RETRIEVAL_SCORE_THRESHOLD = 1.5


def build_prompt(
    context_chunk, user_query
):  # Builds a prompt for the LLM using the provided context chunks and user query.
    context_text = "\n\n".join(
        [f"Chunk {i+1}:{chunk}" for i, chunk in enumerate(context_chunk)]
    )  # Combines context chunks into a single string with numbering.
    prompt = f"""
    You are an AI assistant.
    Use only the content provided in the context to answer the user's query.
    If the answer is not present in the context, then clearly respond with 'Information is not available in the provided context.'
    Respond strictly in the JSON format as shown below:
    {{
        'SUMMARY': '',
        KEY_POINTS: [],
        INSIGHTS: '',
        CONFIDENCE: ''
    }} #Specifies the required JSON response format.
    
    Context:
    {context_text}
    
    Query:
    {user_query}
    """  # Constructs the final prompt with context and query.

    return prompt


def run_llm(
    prompt, llm_client
):  # Sends the constructed prompt to the LLM and retrieves the response.
    response = llm_client.generate(
        prompt
    )  # Assumes llm_client has a generate method to get the response.
    return response.strip()  # Strips any leading/trailing whitespace from the response.


def llm_reasoning(
    retrieved_chunks, llm_client, query
):  # Main function to perform LLM reasoning using retrieved context chunks and user query.

    if not retrieved_chunks:  # Checks if there are no retrieved chunks.
        return {
            "SUMMARY": "No relevant information was found.",
            "KEY_POINTS": [],
            "INSIGHTS": "Information is not available in the provided context.",
            "CONFIDENCE": "Low",
        }

    best_score = retrieved_chunks[0]["score"]
    if best_score > RETRIEVAL_SCORE_THRESHOLD:
        return {
            "Summary": "Sufficient relevant information not found in the uploaded document.",
            "Key_Points": [],
            "Insights": "Information is not available in the provided context.\n",
            "Confidence": "Low\n",
        }

    context_chunks = [item["chunk_text"] for item in retrieved_chunks]
    prompt = build_prompt(
        context_chunks, query
    )  # Builds the prompt using the retrieved chunks and user query.
    llm_response = run_llm(
        prompt, llm_client
    )  # Sends the prompt to the LLM and gets the response.

    return llm_response

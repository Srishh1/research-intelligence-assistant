import requests
import json


def build_context(fused_results: list[dict], chunks: list[dict]) -> str:
    """
    Takes the top fused results and builds a context string
    for the LLM. We look up the full abstract for each result.
    """

    # Build a lookup from title to full chunk text
    chunk_lookup = {chunk["metadata"]["title"]: chunk["text"] for chunk in chunks}

    context_parts = []

    for result in fused_results:
        title = result["title"]
        link = result["link"]
        full_text = chunk_lookup.get(title, f"Title: {title}")

        context_parts.append(f"""
PAPER {result['rank']}:
{full_text}
Link: {link}
Confidence: found in {', '.join(result['found_in'])}
""")

    return "\n---\n".join(context_parts)


def ask_llm(query: str, context: str) -> str:
    """
    Sends query + context to Mistral via Groq API.
    Groq is free, fast (2-3 seconds), and works in the cloud.
    """
    import os
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        raise Exception("GROQ_API_KEY environment variable not set")

    system_prompt = """You are a research assistant helping AI/ML students find relevant papers and identify research opportunities.

You will be given a student's question and a set of relevant research papers fetched from ArXiv.

Your job is to:
1. Directly answer the student's question using the papers provided
2. Highlight the most important papers and what they contribute
3. Identify open problems or research gaps mentioned in the papers
4. Suggest a concrete direction the student could take for their own research

Be specific, cite paper titles, and be helpful to someone planning to publish their first research paper."""

    user_message = f"""Student Question: {query}

Relevant Papers:
{context}

Please provide a comprehensive answer that helps the student understand the landscape and find a research direction."""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
    )

    if response.status_code != 200:
        raise Exception(f"Groq API failed: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def run_pipeline(
    query: str,
    index,
    chunks: list[dict],
    graph,
    top_k: int = 5
) -> dict:
    """
    Runs the full HybridRAG pipeline end to end.
    This is the single function your API and UI will call.
    """
    from retrieval.vector_store import vector_search
    from retrieval.graph_store import graph_search
    from retrieval.fusion import reciprocal_rank_fusion

    print(f"\nRunning pipeline for: '{query}'")

    # Step 1: Retrieve from both sources
    vector_results = vector_search(query, index, chunks, top_k=top_k)
    graph_results = graph_search(query, graph, chunks, top_k=top_k)

    # Step 2: Fuse
    fused_results = reciprocal_rank_fusion(vector_results, graph_results, top_k=top_k)

    # Step 3: Build context
    context = build_context(fused_results, chunks)

    # Step 4: Ask LLM
    print("Asking groq...")
    answer = ask_llm(query, context)

    return {
        "query": query,
        "answer": answer,
        "sources": fused_results
    }
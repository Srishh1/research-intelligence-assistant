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
    Sends the query + context to Mistral via Ollama
    and returns the answer.
    """

    # This is the prompt that instructs the LLM how to behave
    system_prompt = """You are a research assistant helping AI/ML students find relevant papers and identify research opportunities.

You will be given a student's question and a set of relevant research papers fetched from ArXiv.

Your job is to:
1. Directly answer the student's question using the papers provided
2. Highlight the most important papers and what they contribute
3. Identify open problems or research gaps mentioned in the papers
4. Suggest a concrete direction the student could take for their own research

Be specific, cite paper titles, and be helpful to someone planning to publish their first research paper."""

    # Format the full prompt
    user_message = f"""Student Question: {query}

Relevant Papers:
{context}

Please provide a comprehensive answer that helps the student understand the landscape and find a research direction."""

    # Call Ollama's API — it runs locally on port 11434
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "mistral",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False  # Wait for full response
        }
    )

    if response.status_code != 200:
        raise Exception(f"Ollama API failed: {response.status_code} - {response.text}")

    data = response.json()
    return data["message"]["content"]


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
    print("Asking Mistral...")
    answer = ask_llm(query, context)

    return {
        "query": query,
        "answer": answer,
        "sources": fused_results
    }
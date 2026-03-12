from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the embedding model once at module level
# This model converts text into 384-dimensional vectors
# "all-MiniLM-L6-v2" is small, fast, and surprisingly good
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_vector_store(chunks: list[dict]) -> tuple:
    """
    Takes chunks, embeds them, and stores in FAISS.
    Returns the FAISS index and the original chunks
    (we need chunks to retrieve metadata later).
    """

    # Extract just the text strings for embedding
    texts = [chunk["text"] for chunk in chunks]

    print("Embedding papers... this takes a few seconds.")

    # Convert texts to vectors
    # Shape will be (num_chunks, 384)
    embeddings = model.encode(texts, show_progress_bar=True)

    # Convert to float32 — FAISS requires this specific type
    embeddings = np.array(embeddings, dtype="float32")

    # Get the vector dimension (384 for this model)
    dimension = embeddings.shape[1]

    # Create a FAISS index
    # IndexFlatL2 means "find vectors with smallest L2 distance"
    # L2 distance = how far apart two vectors are = how different two texts are
    index = faiss.IndexFlatL2(dimension)

    # Add all our embeddings to the index
    index.add(embeddings)

    print(f"Vector store built with {index.ntotal} vectors.")

    return index, chunks


def vector_search(query: str, index, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Takes a query string, embeds it, and finds the most
    similar chunks in the FAISS index.
    """

    # Embed the query the same way we embedded the chunks
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    # Search the index — returns distances and indices of top_k results
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i, idx in enumerate(indices[0]):
        # idx is the position in our chunks list
        chunk = chunks[idx]

        # Add the similarity score to the result
        # Lower distance = more similar, so we invert it for intuition
        result = {
            "rank": i + 1,
            "score": float(distances[0][i]),
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        }
        results.append(result)

    return results
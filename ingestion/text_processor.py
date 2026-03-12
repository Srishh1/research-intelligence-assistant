def papers_to_chunks(papers: list[dict]) -> list[dict]:
    """
    Takes a list of paper dictionaries and converts them into
    searchable chunks. Each chunk carries metadata so we always
    know which paper it came from.
    """

    chunks = []

    for paper in papers:

        # We combine title + abstract as one chunk
        # This is the core searchable text for each paper
        text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"

        # Each chunk is a dict with the text AND metadata
        # Metadata is crucial — when we retrieve a chunk later,
        # we need to know which paper it belongs to
        chunk = {
            "text": text,
            "metadata": {
                "title": paper["title"],
                "authors": paper["authors"],
                "published": paper["published"],
                "link": paper["link"]
            }
        }

        chunks.append(chunk)

    return chunks


def get_texts_for_embedding(chunks: list[dict]) -> list[str]:
    """
    Extracts just the raw text strings from chunks.
    This is what gets fed into the embedding model.
    """
    return [chunk["text"] for chunk in chunks]
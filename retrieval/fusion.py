def reciprocal_rank_fusion(
    vector_results: list[dict],
    graph_results: list[dict],
    top_k: int = 5,
    k: int = 60
) -> list[dict]:
    """
    Merges vector search results and graph search results
    into a single ranked list using Reciprocal Rank Fusion.

    k=60 is the standard constant from the original RRF paper.
    It smooths out the impact of very high rankings.
    """

    # This dict will hold the fused score for each paper
    # Key = paper title, Value = combined score + metadata
    fused_scores = {}

    # Process vector search results
    for result in vector_results:
        title = result["metadata"]["title"]
        rank = result["rank"]

        # RRF formula
        rrf_score = 1 / (rank + k)

        if title not in fused_scores:
            fused_scores[title] = {
                "title": title,
                "link": result["metadata"]["link"],
                "authors": result["metadata"]["authors"],
                "published": result["metadata"]["published"],
                "rrf_score": 0,
                "found_in": []
            }

        fused_scores[title]["rrf_score"] += rrf_score
        fused_scores[title]["found_in"].append(f"vector(rank {rank})")

    # Process graph search results
    for result in graph_results:
        title = result["title"]
        rank = result["rank"]

        rrf_score = 1 / (rank + k)

        if title not in fused_scores:
            # This paper was found by graph but not vector search
            fused_scores[title] = {
                "title": title,
                "link": result["link"],
                "authors": [],
                "published": "",
                "rrf_score": 0,
                "found_in": []
            }

        fused_scores[title]["rrf_score"] += rrf_score
        fused_scores[title]["found_in"].append(f"graph(rank {rank})")

    # Sort by fused score, highest first
    sorted_results = sorted(
        fused_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )[:top_k]

    # Add final rank
    for i, result in enumerate(sorted_results):
        result["rank"] = i + 1

    return sorted_results
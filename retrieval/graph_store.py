import spacy
import networkx as nx

# Load spaCy's English model
# en_core_web_sm can extract entities like people, orgs, concepts
nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> list[str]:
    """
    Improved entity extraction that focuses on technical terms
    rather than generic noun phrases.
    """
    doc = nlp(text)

    entities = set()

    # Named entities only from useful categories
    useful_labels = {"ORG", "PRODUCT", "GPE", "WORK_OF_ART", "EVENT", "LAW"}
    for ent in doc.ents:
        if ent.label_ in useful_labels:
            cleaned = ent.text.strip().lower()
            if len(cleaned) > 3:
                entities.add(cleaned)

    # Technical noun chunks — filter out generic filler words
    stopwords = {
        "what", "this", "that", "these", "those", "which", "their",
        "our", "we", "they", "it", "its", "the", "a", "an",
        "paper", "work", "study", "approach", "method", "result",
        "problem", "task", "model", "system", "performance"
    }

    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip().lower()
        # Skip short, generic, or stopword-only chunks
        if len(cleaned) > 4 and cleaned not in stopwords:
            words = cleaned.split()
            if not all(w in stopwords for w in words):
                entities.add(cleaned)

    return list(entities)


def build_knowledge_graph(chunks: list[dict]) -> nx.Graph:
    """
    Builds a NetworkX graph from paper chunks.
    Each entity becomes a node.
    Entities that appear in the same paper get connected by an edge.
    The edge also stores which paper they co-appeared in.
    """

    G = nx.Graph()

    for chunk in chunks:
        title = chunk["metadata"]["title"]
        link = chunk["metadata"]["link"]
        text = chunk["text"]

        # Extract entities from this paper
        entities = extract_entities(text)

        # Add each entity as a node
        # We store the paper info on the node itself
        for entity in entities:
            if not G.has_node(entity):
                G.add_node(entity, papers=[])
            G.nodes[entity]["papers"].append({
                "title": title,
                "link": link
            })

        # Connect every pair of entities that appear in this paper
        # This is what creates the "relationship" between concepts
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                if G.has_edge(e1, e2):
                    # Edge exists — just add this paper to its list
                    G[e1][e2]["papers"].append(title)
                else:
                    # New edge
                    G.add_edge(e1, e2, papers=[title])

    print(f"Knowledge graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def graph_search(query: str, graph: nx.Graph, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Improved graph search with better entity matching.
    """

    query_entities = extract_entities(query)
    query_words = set(query.lower().split())

    print(f"Query entities found: {query_entities}")

    # Match nodes using three strategies
    matched_nodes = set()

    for node in graph.nodes():
        node_lower = node.lower()
        node_words = set(node_lower.split())

        # Strategy 1: direct entity match
        for qe in query_entities:
            if qe in node_lower or node_lower in qe:
                matched_nodes.add(node)

        # Strategy 2: word overlap with query words
        # If 2+ words from query appear in a node, it's relevant
        overlap = query_words & node_words
        meaningful_overlap = overlap - {
            "what", "are", "the", "in", "and", "where", "can",
            "i", "a", "an", "of", "for", "to", "is", "how"
        }
        if len(meaningful_overlap) >= 2:
            matched_nodes.add(node)

        # Strategy 3: key technical terms
        key_terms = ["rag", "retrieval", "generation", "augmented",
                     "llm", "embedding", "vector", "knowledge", "graph"]
        for term in key_terms:
            if term in node_lower:
                matched_nodes.add(node)

    print(f"Matched {len(matched_nodes)} graph nodes")

    # Expand to neighbours
    expanded_nodes = set(matched_nodes)
    for node in list(matched_nodes):
        if node in graph:
            neighbours = list(graph.neighbors(node))
            expanded_nodes.update(neighbours[:3])

    # Score papers by how many matched nodes reference them
    paper_scores = {}
    for node in expanded_nodes:
        if node not in graph:
            continue
        for paper in graph.nodes[node].get("papers", []):
            title = paper["title"]
            if title not in paper_scores:
                paper_scores[title] = {"score": 0, "title": title, "link": paper["link"]}
            paper_scores[title]["score"] += 1

    sorted_papers = sorted(
        paper_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]

    return [{"rank": i+1, "score": p["score"], "title": p["title"], "link": p["link"]}
            for i, p in enumerate(sorted_papers)]
import networkx as nx
from pyvis.network import Network
import os


def build_paper_graph(chunks: list[dict], graph: nx.Graph) -> str:
    """
    Builds a simplified paper-to-paper graph for visualization.
    Instead of showing all 400+ entity nodes, we show:
    - Paper nodes (one per paper)
    - Edges between papers that share important concepts
    - Edge weight = number of shared concepts

    Returns the HTML string of the interactive visualization.
    """

    # Build a paper-level graph
    # Two papers are connected if they share entities in the knowledge graph
    paper_graph = nx.Graph()

    # Add one node per paper
    for chunk in chunks:
        title = chunk["metadata"]["title"]
        link = chunk["metadata"]["link"]
        authors = chunk["metadata"]["authors"]
        published = chunk["metadata"]["published"][:4]  # just the year

        # Shorten title for display
        short_title = title[:50] + "..." if len(title) > 50 else title

        paper_graph.add_node(
            title,
            label=short_title,
            link=link,
            authors=", ".join(authors[:2]),
            year=published,
            size=25
        )

    # Find shared entities between papers
    # For each entity node in the knowledge graph,
    # if it appears in 2+ papers, connect those papers
    for node in graph.nodes():
        papers_with_node = graph.nodes[node].get("papers", [])

        if len(papers_with_node) < 2:
            continue

        # Connect every pair of papers that share this entity
        titles = [p["title"] for p in papers_with_node]
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                t1, t2 = titles[i], titles[j]

                if not paper_graph.has_node(t1) or not paper_graph.has_node(t2):
                    continue

                if paper_graph.has_edge(t1, t2):
                    # Increment shared concept count
                    paper_graph[t1][t2]["weight"] += 1
                else:
                    paper_graph.add_edge(t1, t2, weight=1)

    # Only keep edges with enough shared concepts (reduces clutter)
    edges_to_remove = [
        (u, v) for u, v, d in paper_graph.edges(data=True)
        if d["weight"] < 3
    ]
    paper_graph.remove_edges_from(edges_to_remove)

    # Build pyvis network
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0d1117",  # dark background
        font_color="white",
        notebook=False
    )

    # Physics settings for nice layout
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.005,
                "springLength": 200,
                "springConstant": 0.05
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
                "iterations": 150
            }
        },
        "nodes": {
            "borderWidth": 2,
            "font": {
                "size": 0,
                "color": "rgba(0,0,0,0)"
            },
            "shadow": false
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "shadow": false
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 50,
            "hideEdgesOnDrag": true
        }
    }
    """)
    
    # Add nodes with colors based on connectivity
    degrees = dict(paper_graph.degree())
    max_degree = max(degrees.values()) if degrees else 1

    for node_id, data in paper_graph.nodes(data=True):
        degree = degrees.get(node_id, 0)

        # More connected papers get warmer colors and bigger size
        ratio = degree / max_degree if max_degree > 0 else 0

        if ratio > 0.6:
            color = "#f97316"   # orange — highly connected
        elif ratio > 0.3:
            color = "#8b5cf6"   # purple — moderately connected
        else:
            color = "#3b82f6"   # blue — less connected

        size = 20 + (degree * 5)

        tooltip = (
            f"<b>{data.get('label', node_id)}</b><br>"
            f"Authors: {data.get('authors', 'N/A')}<br>"
            f"Year: {data.get('year', 'N/A')}<br>"
            f"<a href='{data.get('link', '#')}'>Open paper</a>"
        )
        
        net.add_node(
            node_id,
            label=data.get("label", node_id),
            color=color,
            size=size,
            title=tooltip
        )

    # Add edges with thickness based on shared concept count
    for u, v, data in paper_graph.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(
            u, v,
            value=weight,
            title=f"{weight} shared concepts",
            color="#4b5563"
        )

    # Generate HTML
    html = net.generate_html()
    return html
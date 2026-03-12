from ingestion.arxiv_loader import fetch_papers
from ingestion.text_processor import papers_to_chunks
from retrieval.vector_store import build_vector_store
from retrieval.graph_store import build_knowledge_graph
from orchestration.graph_flow import run_pipeline

test_cases = [
    ("vision transformer image classification", 
     "what are the limitations of vision transformers compared to CNNs?"),
    
    ("large language model hallucination", 
     "what techniques exist to reduce hallucinations in LLMs?"),
    
    ("reinforcement learning from human feedback", 
     "what are the open problems in RLHF and reward modeling?"),
    
    ("graph neural network node classification", 
     "what are the scalability challenges in GNNs?"),
    
    ("multimodal large language model vision language", 
     "how do vision language models handle reasoning across modalities?"),
]

for topic, question in test_cases:
    print(f"\n{'='*60}")
    print(f"TOPIC: {topic}")
    print(f"QUESTION: {question}")
    print('='*60)

    papers = fetch_papers(topic, max_results=10)

    if not papers:
        print(f"WARNING: No papers found for topic '{topic}', skipping.")
        continue

    chunks = papers_to_chunks(papers)
    index, chunks = build_vector_store(chunks)
    graph = build_knowledge_graph(chunks)

    result = run_pipeline(
        query=question,
        index=index,
        chunks=chunks,
        graph=graph
    )

    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  {s['rank']}. {s['title']}")
        print(f"     Found in: {', '.join(s['found_in'])}")

    print("\nANSWER PREVIEW:")
    print(result["answer"][:400], "...")
    print()
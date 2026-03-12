from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

from ingestion.arxiv_loader import fetch_papers
from ingestion.text_processor import papers_to_chunks
from retrieval.vector_store import build_vector_store
from retrieval.graph_store import build_knowledge_graph
from orchestration.graph_flow import run_pipeline

app = FastAPI(
    title="Research Intelligence Assistant",
    description="HybridRAG system combining vector search and knowledge graphs for research paper discovery",
    version="1.0.0"
)

# In-memory cache — stores built indexes per topic
# So if someone asks two questions on the same topic,
# we don't re-fetch and re-embed everything
cache = {}


class QueryRequest(BaseModel):
    topic: str      # e.g. "retrieval augmented generation"
    question: str   # e.g. "what are the open problems?"
    max_papers: int = 15


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    topic: str
    papers_analyzed: int
    time_taken_seconds: float


@app.get("/")
def root():
    return {"message": "Research Intelligence Assistant is running"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start_time = time.time()

    try:
        # Check cache first — if we already built the index for this topic, reuse it
        if request.topic not in cache:
            print(f"Cache miss — fetching papers for topic: {request.topic}")

            # Fetch papers
            papers = fetch_papers(request.topic, max_results=request.max_papers)

            if not papers:
                raise HTTPException(status_code=404, detail="No papers found for this topic")

            # Build both stores
            chunks = papers_to_chunks(papers)
            index, chunks = build_vector_store(chunks)
            graph = build_knowledge_graph(chunks)

            # Store in cache
            cache[request.topic] = {
                "index": index,
                "chunks": chunks,
                "graph": graph,
                "paper_count": len(papers)
            }

            print(f"Cache built for topic: {request.topic}")
        else:
            print(f"Cache hit for topic: {request.topic}")

        # Pull from cache
        cached = cache[request.topic]

        # Run the pipeline
        result = run_pipeline(
            query=request.question,
            index=cached["index"],
            chunks=cached["chunks"],
            graph=cached["graph"]
        )

        time_taken = round(time.time() - start_time, 2)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            topic=request.topic,
            papers_analyzed=cached["paper_count"],
            time_taken_seconds=time_taken
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}


@app.get("/cache/topics")
def list_cached_topics():
    return {"cached_topics": list(cache.keys())}
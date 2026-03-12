import requests
import xml.etree.ElementTree as ET

def fetch_papers(topic: str, max_results: int = 20) -> list[dict]:
    """
    Fetches papers from ArXiv using a smart query strategy.
    Tries exact phrase first, falls back to keyword search if needed.
    """

    def make_request(query_str):
        url = f"https://export.arxiv.org/api/query?search_query={query_str}&max_results={max_results}&sortBy=relevance"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"ArXiv API failed with status {response.status_code}")
        return response

    def parse_papers(response):
        root = ET.fromstring(response.content)
        namespace = "{http://www.w3.org/2005/Atom}"
        papers = []
        for entry in root.findall(f"{namespace}entry"):
            title = entry.find(f"{namespace}title").text.strip()
            abstract = entry.find(f"{namespace}summary").text.strip()
            published = entry.find(f"{namespace}published").text.strip()
            link = entry.find(f"{namespace}id").text.strip()
            authors = [
                author.find(f"{namespace}name").text
                for author in entry.findall(f"{namespace}author")
            ]
            papers.append({
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published,
                "link": link
            })
        return papers

    print(f"Fetching papers on: {topic}...")

    # Strategy 1: exact phrase search
    phrase = topic.replace(" ", "+")
    exact_query = f"all:%22{phrase}%22"
    response = make_request(exact_query)
    papers = parse_papers(response)

    # Strategy 2: if exact phrase returns too few, fall back to keyword search
    if len(papers) < 5:
        print(f"Exact phrase returned {len(papers)} results, trying keyword search...")
        keyword_query = "all:" + "+".join(topic.split())
        response = make_request(keyword_query)
        papers = parse_papers(response)

    # Strategy 3: if still too few, try title-only search with first two words
    if len(papers) < 5:
        print(f"Keyword search returned {len(papers)} results, trying title search...")
        first_two = "+".join(topic.split()[:2])
        title_query = f"ti:{first_two}"
        response = make_request(title_query)
        papers = parse_papers(response)

    print(f"Found {len(papers)} papers.")
    return papers
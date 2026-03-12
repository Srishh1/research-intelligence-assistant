import gradio as gr
import requests
import json

API_URL = "http://127.0.0.1:8000"


def query_research_assistant(topic: str, question: str, max_papers: int):
    """
    Calls the FastAPI backend and formats the response for display.
    """

    if not topic.strip():
        return "Please enter a research topic.", "", ""

    if not question.strip():
        return "Please enter a question.", "", ""

    try:
        # Call our FastAPI backend
        response = requests.post(
            f"{API_URL}/query",
            json={
                "topic": topic,
                "question": question,
                "max_papers": int(max_papers)
            },
            timeout=120  # 2 minute timeout for slow queries
        )

        if response.status_code != 200:
            return f"Error: {response.text}", "", ""

        data = response.json()

        # Format the answer
        answer = data["answer"]

        # Format sources as a readable string
        sources_text = f"**Papers Analyzed:** {data['papers_analyzed']} | **Time:** {data['time_taken_seconds']}s\n\n"
        for source in data["sources"]:
            sources_text += f"**{source['rank']}. {source['title']}**\n"
            sources_text += f"🔗 {source['link']}\n"
            if source.get('authors'):
                authors = source['authors'][:3]  # show max 3 authors
                sources_text += f"👤 {', '.join(authors)}\n"
            sources_text += f"📊 Found in: {', '.join(source['found_in'])}\n\n"

        # Format metadata
        meta = f"Topic indexed: `{data['topic']}` | Papers used: {data['papers_analyzed']}"

        return answer, sources_text, meta

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend. Make sure FastAPI is running (`uvicorn api.main:app --reload`)", "", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""


# Build the Gradio interface
with gr.Blocks(
    title="Research Intelligence Assistant",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🔬 Research Intelligence Assistant
    ### Powered by HybridRAG — Vector Search + Knowledge Graph + Mistral AI

    Enter a research topic and ask any question. The system will:
    - Fetch live papers from ArXiv
    - Search semantically using FAISS vector embeddings
    - Traverse a knowledge graph of concepts and entities
    - Fuse both results using Reciprocal Rank Fusion
    - Generate a research-grade answer using Mistral 7B
    """)

    with gr.Row():
        with gr.Column(scale=1):
            topic_input = gr.Textbox(
                label="Research Topic",
                placeholder="e.g. retrieval augmented generation",
                info="Papers will be fetched from ArXiv on this topic"
            )
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What are the open problems and where can I contribute?",
                lines=3
            )
            max_papers_slider = gr.Slider(
                minimum=5,
                maximum=30,
                value=15,
                step=5,
                label="Number of papers to analyze"
            )
            submit_btn = gr.Button("🔍 Search & Analyze", variant="primary")

            gr.Markdown("**Example queries:**")
            gr.Examples(
                examples=[
                    ["retrieval augmented generation", "What are the open problems I can contribute to?"],
                    ["large language model fine tuning", "What methods exist and what are their tradeoffs?"],
                    ["graph neural networks", "What are recent advances and open research gaps?"],
                    ["vision language models", "How do these work and what problems remain unsolved?"],
                ],
                inputs=[topic_input, question_input]
            )

        with gr.Column(scale=2):
            meta_output = gr.Markdown(label="Status")
            answer_output = gr.Markdown(label="Answer")
            sources_output = gr.Markdown(label="Sources")

    submit_btn.click(
        fn=query_research_assistant,
        inputs=[topic_input, question_input, max_papers_slider],
        outputs=[answer_output, sources_output, meta_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
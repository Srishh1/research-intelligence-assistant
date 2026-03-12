import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000"


def query_research_assistant(topic: str, question: str, max_papers: int):
    if not topic.strip():
        return "Please enter a research topic.", "", "", "<p>No graph yet.</p>"
    if not question.strip():
        return "Please enter a question.", "", "", "<p>No graph yet.</p>"

    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "topic": topic,
                "question": question,
                "max_papers": int(max_papers)
            },
            timeout=300
        )

        if response.status_code != 200:
            return f"Error: {response.text}", "", "", "<p>Error</p>"

        data = response.json()

        answer = data["answer"]

        sources_text = f"**Papers analyzed:** {data['papers_analyzed']} | **Time:** {data['time_taken_seconds']}s\n\n"
        for source in data["sources"]:
            sources_text += f"**{source['rank']}. {source['title']}**\n"
            sources_text += f"🔗 {source['link']}\n"
            if source.get("authors"):
                sources_text += f"👤 {', '.join(source['authors'][:2])}\n"
            sources_text += f"📊 Found in: {', '.join(source['found_in'])}\n\n"

        meta = f"Topic: `{data['topic']}` | Papers: {data['papers_analyzed']}"

        topic_url = topic.strip().replace(" ", "%20")
        graph_url = f"http://127.0.0.1:8000/graph/{topic_url}"
        graph_md = f"### [🕸️ Click here to open Citation Graph]({graph_url})\n\nOpens in a new tab — drag nodes, hover for paper details."

        return answer, sources_text, meta, graph_md

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend. Make sure FastAPI is running.", "", "", "<p>No connection</p>"
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "<p>Error</p>"


with gr.Blocks(title="Research Intelligence Assistant", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔬 Research Intelligence Assistant
    ### HybridRAG — Vector Search + Knowledge Graph + Mistral AI
    """)

    with gr.Row():
        with gr.Column(scale=1):
            topic_input = gr.Textbox(
                label="Research Topic",
                placeholder="e.g. retrieval augmented generation"
            )
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What are the open problems?",
                lines=3
            )
            max_papers_slider = gr.Slider(
                minimum=5, maximum=30, value=15, step=5,
                label="Number of papers to analyze"
            )
            submit_btn = gr.Button("🔍 Search & Analyze", variant="primary")

            gr.Examples(
                examples=[
                    ["retrieval augmented generation", "What are the open problems I can contribute to?"],
                    ["large language model hallucination", "What techniques exist to reduce hallucinations?"],
                    ["graph neural networks", "What are recent advances and open research gaps?"],
                    ["vision language models", "How do these work and what problems remain unsolved?"],
                ],
                inputs=[topic_input, question_input]
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("💬 Answer"):
                    meta_output = gr.Markdown()
                    answer_output = gr.Markdown()

                with gr.Tab("📚 Sources"):
                    sources_output = gr.Markdown()

                with gr.Tab("🕸️ Citation Graph"):
                    gr.Markdown("**Paper relationship graph** — nodes are papers, edges show shared concepts. Hover for details, drag to explore.")
                    graph_link = gr.Markdown()


    submit_btn.click(
        fn=query_research_assistant,
        inputs=[topic_input, question_input, max_papers_slider],
        outputs=[answer_output, sources_output, meta_output, graph_link]
    )

if __name__ == "__main__":
    demo.launch(share=False)
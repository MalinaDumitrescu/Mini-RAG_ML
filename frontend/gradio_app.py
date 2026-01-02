import gradio as gr
import requests
import json

API_URL = "http://127.0.0.1:8000/api/v1/chat"

def chat_with_rag(message, history):
    try:
        payload = {"message": message, "history": []}
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        answer = data.get("answer", "No answer received.")
        sources = data.get("sources", [])
        judge = data.get("judge_result")
        
        # Format sources
        sources_text = ""
        if sources:
            sources_text = "\n\n---\n**Sources:**\n"
            for i, s in enumerate(sources, 1):
                # Truncate source text for display
                preview = s[:300].replace("\n", " ") + "..."
                sources_text += f"{i}. {preview}\n"
        
        # Format Judge Result
        judge_text = ""
        if judge:
            verdict = judge.get("verdict", "unknown").upper()
            scores = judge.get("scores", {})
            judge_text = f"\n\n---\n**Judge Verdict:** {verdict}\n"
            judge_text += f"Scores: {json.dumps(scores, indent=2)}"

        full_response = answer + sources_text + judge_text
        return full_response
        
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="ML RAG Course Assistant") as demo:
    gr.Markdown("# 🤖 ML RAG Course Assistant")
    gr.Markdown("Ask questions about the Machine Learning course. The system uses RAG to retrieve context and an LLM Judge to evaluate answers.")
    
    chat_interface = gr.ChatInterface(
        fn=chat_with_rag,
        examples=[
            "What is backpropagation?", 
            "Explain the difference between overfitting and underfitting.",
            "How does a Transformer work?"
        ],
        title="Chat"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

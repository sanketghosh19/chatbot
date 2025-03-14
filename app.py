# app.py

import gradio as gr
from models import get_gemini_response, get_mistral_response, get_deepseek_response
import os

# Load custom CSS from style.css if available
css_file = "style.css"
custom_css = ""
if os.path.exists(css_file):
    with open(css_file, "r") as f:
        custom_css = f.read()

def chat_interface(user_message, model_choice, history):
    """
    Gradio interface function:
      - user_message: The user's input text.
      - model_choice: Either "Gemini", "Mistral", or "Deepseek".
      - history: List of (user, bot) tuples.
      
    Returns:
      - A cleared text input.
      - Updated conversation (list of tuples) for the Chatbot.
      - Updated conversation state.
      - A plain text version of the full chat history (for the sidebar).
    """
    if history is None:
        history = []

    if model_choice == "Gemini":
        bot_response = get_gemini_response(user_message)
    elif model_choice == "Mistral":
        bot_response = get_mistral_response(user_message)
    elif model_choice == "Deepseek":
        bot_response = get_deepseek_response(user_message)
    else:
        bot_response = "Model not supported."

    history.append((user_message, bot_response))

    # Prepare plain text history for the sidebar
    history_text = ""
    for i, (user, bot) in enumerate(history, start=1):
        history_text += f"Q{i}: {user}\nA{i}: {bot}\n\n"

    return "", history, history, history_text

with gr.Blocks(css=custom_css) as demo:
    # Centered header
    gr.Markdown("<h2>Multi-Model LLM Chatbot</h2>")

    with gr.Row():
        # Left Column: Sidebar for Chat History
        with gr.Column(scale=1):
            with gr.Accordion("Chat History", open=False):
                history_box = gr.Textbox(
                    label="Conversation Log",
                    lines=15,
                    interactive=False
                )
        # Right Column: Main Chat Interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat")
            state = gr.State([])

            # Dropdown for selecting the LLM model (now includes Deepseek)
            model_choice = gr.Dropdown(
                label="Choose LLM Model",
                choices=["Gemini", "Mistral", "Deepseek"],
                value="Gemini"
            )

            user_input = gr.Textbox(
                show_label=False,
                placeholder="Type your message here...",
                lines=2
            )
            send_button = gr.Button("Send")

            # Submit on pressing Enter
            user_input.submit(
                fn=chat_interface,
                inputs=[user_input, model_choice, state],
                outputs=[user_input, chatbot, state, history_box]
            )
            # Submit on clicking the "Send" button
            send_button.click(
                fn=chat_interface,
                inputs=[user_input, model_choice, state],
                outputs=[user_input, chatbot, state, history_box]
            )

demo.launch()

import gradio as gr
import os
import time
import moe 

user_input = {
    "instruction": None,
    "methodIn": None,
    "alphaIn": 16,
    "expertsK": 3,
    "continue_prompt": "yes"
}

def get_model_output(chat_history, user_message):
    instruction = user_message['message']['content']
    methodIn = "transformer" 
    weights = moe.get_weights(instruction, methodIn)
    alphas = moe.mult_weights_by_alpha(weights, int(user_input["alphaIn"]), int(user_input["expertsK"]))
    output = moe.generate_output(instruction, moe.model, alphas, moe.tokenizer, moe.generation_args)
    chat_history.append({
        "role": "system",
        "content": f"MoE Model: {output}"
    })
    output_base = moe.generate_base_output(instruction, moe.base_model, alphas, moe.base_tokenizer, moe.generation_args)
    chat_history.append({
        "role": "system",
        "content": f"Base Model: {output_base}"
    })
    return chat_history

def create_chat_interface():
    with gr.Blocks() as chat_tab:
        gr.Markdown("# Hydra MoE Web Interface")
        with gr.Row():
            chat_display = gr.Chatbot([], elem_id="chatbot")
        chat_input = gr.Textbox("")
        chat_input.submit(
            get_model_output,
            inputs=[chat_display, chat_input],
            outputs=[chat_display],
        )
    return chat_tab

def init_interface():
    chat_interface = create_chat_interface()
    chat_interface.launch(server_name="0.0.0.0", server_port=8001, inbrowser=True)


def main():
    init_interface()
    while True:
        time.sleep(0.5)


if __name__ == "__main__":
    main()
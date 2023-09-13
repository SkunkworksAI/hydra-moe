import moe 
import gradio as gr
import os
import time

from dataclasses import dataclass
from typing import Literal
import logging 
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    method: Literal['combined','transformer','centroid'] = 'transformer'
    alphaIn : int = 16
    expertsK : int = 2

default_config = InferenceConfig()

def get_model_output(moe_chat_history, base_chat_history, instruction, method, expertsK, max_tokens):
    logger.info(moe_chat_history)
    logger.info(base_chat_history)
    logger.info(instruction)
    
    weights = moe.get_weights(instruction, method)
    logger.info(weights)
    
    alphas = moe.mult_weights_by_alpha(weights, int(default_config.alphaIn), int(expertsK))
    logger.info(alphas)
    
    output = moe.generate_output(instruction, moe.model, alphas, moe.tokenizer, moe.generation_args, count=max_tokens)
    logger.info(output)
    
    response = output.split("### Response:\n")[1].split("</s>")[0].strip()
    moe_chat_history.append((instruction, f"MoE Model: {response}"))
    
    output_base = moe.generate_base_output(instruction, moe.base_model, alphas, moe.base_tokenizer, moe.generation_args, count=max_tokens)
    
    response_base = output_base.split("### Response:\n")[1].split("</s>")[0].strip()
    base_chat_history.append((instruction, f"Base Model: {response_base}"))
    
    return moe_chat_history, base_chat_history


def create_chat_interface():
    with gr.Blocks() as chat_interface:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Hydra MoE")
                moe_chat_display = gr.Chatbot([], elem_id="moe_chat")
            with gr.Column():
                gr.Markdown("# Base")
                base_chat_display = gr.Chatbot([], elem_id="base_chat")
        with gr.Row():
            method_dropdown = gr.Dropdown(["combined", "transformer", "centroid"], label="Method", value="transformer")
            expertsK_slider = gr.Slider(minimum=1, maximum=10, value=3, label="Experts K")
            max_tokens_slider = gr.Slider(minimum=50, maximum=1024, value=320, label="Max Tokens")
        chat_input = gr.Textbox("Tell me a story about a hydra.")
        
        chat_input.submit(
            get_model_output,
            inputs=[moe_chat_display, base_chat_display, chat_input, method_dropdown, expertsK_slider, max_tokens_slider],
            outputs=[moe_chat_display, base_chat_display],
        )
    return chat_interface

def init_interface():
    with gr.Blocks() as interface:
        with gr.Tab("Inference A/B"):
            create_chat_interface()
        with gr.Tab("Finetune"):
            gr.Markdown("Finetuning Interface Stub")
    logger.info("Interface initialized")
    interface.launch(server_name="0.0.0.0", server_port=8001, inbrowser=True)

def main():
    logger.info("Calling main")
    


def init_interface():
    chat_interface = create_chat_interface()
    logger.info("Chat interfaced initialized")
    chat_interface.launch(server_name="0.0.0.0", server_port=8001, inbrowser=True)


def main():
    logger.info("Calling main")
    
    logger.info("Initializing model")
    init_interface()
    while True:
        time.sleep(0.5)


if __name__ == "__main__":
    moe.initialize_model()
    main()
import moe 
import gradio as gr
import time
import pandas as pd

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

def create_stats_df(weights) -> pd.DataFrame:
    df = pd.DataFrame({
    'Expert ID': list(weights.keys()),
    'Weight': [float(val.cpu().detach().numpy()) for val in weights.values()]
    })
    logger.info(df.to_markdown())

    total_weight = df['Weight'].sum()
    df['Weight Percentage (%)'] = (df['Weight'] / total_weight) * 100

    df = df.sort_values(by='Weight Percentage (%)', ascending=False)
    return df

def get_model_output(moe_chat_history, base_chat_history, stats_chat_history, instruction, method, expertsK, max_tokens):
    logger.info(moe_chat_history)
    logger.info(base_chat_history)
    logger.info(stats_chat_history)
    logger.info(instruction)
    
    weights = moe.get_weights(instruction, method)
    logger.info(f"Weights: {weights}")
    
    alphas = moe.mult_weights_by_alpha(weights, int(default_config.alphaIn), int(expertsK))
    logger.info(f"Alphas: {alphas}")
    combined_df = create_stats_df(weights)
    combined_md = combined_df.head(expertsK).to_markdown(index=False)
    logger.info(f"\n{combined_md}")
    stats_chat_history.append((instruction, combined_md))

    output = moe.generate_output(instruction, moe.model, alphas, moe.tokenizer, moe.generation_args, count=max_tokens)
    logger.info(output)
    
    response = output.split("### Response:\n")[1].split("</s>")[0].strip()
    moe_chat_history.append((instruction, f"MoE Model ({method}): {response}"))
    
    output_base = moe.generate_base_output(instruction, moe.base_model, alphas, moe.base_tokenizer, moe.generation_args, count=max_tokens)
    response_base = output_base.split("### Response:\n")[1].split("</s>")[0].strip()
    base_chat_history.append((instruction, f"Base Model ({method}): {response_base}"))
    
    return moe_chat_history, base_chat_history, stats_chat_history

def create_chat_interface():
    with gr.Blocks() as chat_interface:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Hydra MoE")
                moe_chat_display = gr.Chatbot([], elem_id="moe_chat")
            with gr.Column():
                gr.Markdown("# Base")
                base_chat_display = gr.Chatbot([], elem_id="base_chat")
            with gr.Column():
                gr.Markdown("# Stats")
                stats_chat_display = gr.Chatbot([], elem_id="stats_chat")
        with gr.Row():
            method_dropdown = gr.Dropdown(["combined", "transformer", "centroid"], label="Method", value="transformer")
            expertsK_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Experts K")
            max_tokens_slider = gr.Slider(minimum=50, maximum=1024, value=320, label="Max Tokens")
        chat_input = gr.Textbox("Tell me a story about a hydra.")
        
        chat_input.submit(
            get_model_output,
            inputs=[moe_chat_display, base_chat_display, stats_chat_display, chat_input, method_dropdown, expertsK_slider, max_tokens_slider],
            outputs=[moe_chat_display, base_chat_display, stats_chat_display],
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

def config_logger(logger: logging.Logger):
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter('%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_format)

    logger.addHandler(ch)


def main():
    config_logger(logger)
    logger.info("Calling main")
    
    logger.info("Initializing model")
    init_interface()
    while True:
        time.sleep(0.5)


if __name__ == "__main__":
    moe.initialize_model()
    main()
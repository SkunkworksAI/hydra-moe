# If running in WSL you might need to use export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/wsl/lib
# 

# pip install git+https://github.com/vllm-project/vllm
# pip install fschat
python -m vllm.entrypoints.openai.api_server --model SkunkworksAI/Mistralic-7B-1
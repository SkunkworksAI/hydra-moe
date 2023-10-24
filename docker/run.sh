#!/bin/bash
echo "Container start"
python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('$HUGGINGFACE_API_TOKEN')"
#if [ ! -d "gating_v3" ]; then
#    git lfs clone https://huggingface.co/HydraLM/gating_v3
#fi
python3 main.py --inference
echo "Container idle.."
tail -f /dev/null

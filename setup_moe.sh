#!/bin/bash
pip install -r requirements.txt
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/HydraLM/gating_v2 router_files

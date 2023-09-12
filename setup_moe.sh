pip install -r requirements.txt
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install
git lfs clone https://huggingface.co/HydraLM/gating_v2


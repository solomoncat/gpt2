# Remove from sesame before running
python3.11 -m venv .venv
source .venv/bin/activate

upgrade pip
pip install --upgrade pip

install dependencies
pip install transformers==4.47.0 torch
pip install torchvision==0.22.0
pip install torch==2.7.1
pip install flash-attn==2.8.1 --no-build-isolation
pip install datasets wandb accelerate hf_transfer

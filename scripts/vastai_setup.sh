#!/bin/bash
# Vast.ai one-shot setup for Atlas training
# Run this ON the Vast.ai instance after cloning the repo
#
# Usage: bash scripts/vastai_setup.sh

set -euo pipefail

echo "════════════════════════════════════════"
echo "  Atlas — Vast.ai Setup"
echo "════════════════════════════════════════"
echo ""

cd /workspace/atlas

# ── 1. System deps ──
echo ">>> Installing system packages..."
apt-get update -qq && apt-get install -y -qq git python3-pip python3-venv tmux > /dev/null 2>&1
echo "    Done."

# ── 2. Python venv ──
echo ">>> Creating Python venv..."
python3 -m venv .venv
source .venv/bin/activate

# ── 3. Install PyTorch with CUDA 12.4 ──
echo ">>> Installing PyTorch (CUDA 12.4)..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu124

# ── 4. Install remaining deps (skip torch since we just installed it) ──
echo ">>> Installing project dependencies..."
pip install -q tokenizers sentencepiece datasets tqdm wandb pyyaml fastapi uvicorn numpy

# ── 5. Verify GPU ──
echo ""
echo ">>> GPU Check:"
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'    GPU:  {name}')
    print(f'    VRAM: {vram:.1f} GB')
    print(f'    CUDA: {torch.version.cuda}')
else:
    print('    ERROR: CUDA not available!')
    print('    Make sure you picked a PyTorch+CUDA template when renting.')
    exit(1)
"

echo ""
echo "════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps (copy-paste these):"
echo ""
echo "  # 1. Start a tmux session (keeps training alive if SSH drops)"
echo "  tmux new -s train"
echo ""
echo "  # 2. Activate venv"
echo "  source .venv/bin/activate"
echo ""
echo "  # 3. Download data (~1-2 hours for 20GB)"
echo "  python scripts/download_data.py --output data/raw/ --max-size-gb 20"
echo ""
echo "  # 4. Train tokenizer (~5 min)"
echo "  python src/tokenizer/trainer.py --data-dir data/raw/ --output data/tokenizer.json"
echo ""
echo "  # 5. Tokenize data into binary (~10 min)"
echo "  python src/data/preprocessing.py --data-dir data/raw/ --tokenizer data/tokenizer.json --output data/processed/train.pt"
echo ""
echo "  # 6. Start training"
echo "  python scripts/train.py --phase pretrain --precision bf16 --batch-size 32 --grad-accum 2 --save-interval 5000"
echo ""
echo "  To detach from tmux: Ctrl+B then D"
echo "  To reattach: tmux attach -t train"
echo "════════════════════════════════════════"

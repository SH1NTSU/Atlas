# ============================================================
# Atlas — Windows Setup & Training Script (PowerShell)
# ============================================================
# Run this in PowerShell as Administrator (right-click -> Run as Admin)
# Copy-paste sections one at a time, don't run everything at once.
# ============================================================

# ────────────────────────────────────────────────────────────
# STEP 0: Check prerequisites
# ────────────────────────────────────────────────────────────

Write-Host "=== Checking prerequisites ===" -ForegroundColor Cyan

# Check Python
python --version
# If this fails, download Python 3.10+ from https://www.python.org/downloads/
# IMPORTANT: Check "Add Python to PATH" during installation

# Check NVIDIA driver & CUDA
nvidia-smi
# If this fails, install latest GPU driver from https://www.nvidia.com/drivers/
# Note the CUDA version shown (top right) — you need 12.x

# ────────────────────────────────────────────────────────────
# STEP 1: Navigate to project folder
# ────────────────────────────────────────────────────────────

# Change this path to wherever you copied the atlas folder:
cd C:\Users\YourUsername\atlas

# ────────────────────────────────────────────────────────────
# STEP 2: Create virtual environment and activate it
# ────────────────────────────────────────────────────────────

python -m venv .venv
.venv\Scripts\Activate.ps1

# If you get "execution policy" error, run this first:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# ────────────────────────────────────────────────────────────
# STEP 3: Install PyTorch with CUDA
# ────────────────────────────────────────────────────────────

# For CUDA 12.4 (most common with recent drivers):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# If your nvidia-smi shows CUDA 12.1, use this instead:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# If your nvidia-smi shows CUDA 11.8, use this instead:
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# ────────────────────────────────────────────────────────────
# STEP 4: Install remaining dependencies
# ────────────────────────────────────────────────────────────

pip install -r requirements.txt

# ────────────────────────────────────────────────────────────
# STEP 5: Verify GPU is detected
# ────────────────────────────────────────────────────────────

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1), 'GB')"

# Should print:
#   CUDA available: True
#   GPU: NVIDIA GeForce RTX 3060 Ti
#   VRAM: 8.0 GB

# ────────────────────────────────────────────────────────────
# STEP 6: Run tests to verify everything works
# ────────────────────────────────────────────────────────────

python -m pytest tests/ -v --tb=short

# All 89 tests should pass. Do NOT continue if tests fail.
# If tests fail, check the error message and fix before proceeding.

# ────────────────────────────────────────────────────────────
# STEP 7: Download training data
# ────────────────────────────────────────────────────────────

# This downloads filtered code (TypeScript, Go, Rust, Shell, configs)
# from The Stack v2. Adjust --max-size-gb based on your disk space.
# 10GB of raw code is a good starting point.
# This will take a while depending on your internet speed.

python scripts/download_data.py --output data/raw/ --max-size-gb 10

# ────────────────────────────────────────────────────────────
# STEP 8: Train the tokenizer
# ────────────────────────────────────────────────────────────

python -c "import sys; sys.path.insert(0, 'src'); from tokenizer.trainer import train_tokenizer; train_tokenizer('data/raw/', 'data/tokenizer.json', vocab_size=32000)"

# ────────────────────────────────────────────────────────────
# STEP 9: Preprocess data (tokenize into binary format)
# ────────────────────────────────────────────────────────────

python -c "import sys; sys.path.insert(0, 'src'); from data.preprocessing import tokenize_and_save; tokenize_and_save('data/raw/', 'data/tokenizer.json', 'data/processed/train.pt', seq_length=1024)"

# ────────────────────────────────────────────────────────────
# STEP 10: START TRAINING
# ────────────────────────────────────────────────────────────

# Before training:
#   - Plug in your PC (don't run on battery)
#   - Disable sleep: Settings -> System -> Power -> Screen and sleep -> Never
#   - Close other GPU-heavy apps (games, browsers with hardware acceleration)

python scripts/train.py --phase pretrain --precision fp16

# Training will take ~3-5 days.
# Checkpoints save every 5000 steps automatically.
# You can safely close the window — just resume later (see below).

# ────────────────────────────────────────────────────────────
# RESUME TRAINING (if you stopped or it crashed)
# ────────────────────────────────────────────────────────────

# First, activate venv again:
# .venv\Scripts\Activate.ps1

# Find your latest checkpoint:
# dir checkpoints/

# Resume from it:
# python scripts/train.py --phase pretrain --precision fp16 --resume checkpoints/step_5000

# ────────────────────────────────────────────────────────────
# STEP 11: INSTRUCTION FINE-TUNING (after pretraining finishes)
# ────────────────────────────────────────────────────────────

# python scripts/train.py --phase instruct --precision fp16 --resume checkpoints/final

# ────────────────────────────────────────────────────────────
# STEP 12: TEST YOUR MODEL
# ────────────────────────────────────────────────────────────

# python -c "import sys; sys.path.insert(0, 'src'); from inference.server import AtlasInference, run_cli; engine = AtlasInference('checkpoints/final', 'data/tokenizer.json'); run_cli(engine)"

# ────────────────────────────────────────────────────────────
# TROUBLESHOOTING
# ────────────────────────────────────────────────────────────

# ERROR: "CUDA out of memory"
#   -> Reduce batch size:
#   python scripts/train.py --phase pretrain --precision fp16 --batch-size 4
#   -> Still OOM? Try batch-size 2 with more gradient accumulation:
#   python scripts/train.py --phase pretrain --precision fp16 --batch-size 2 --grad-accum 16

# ERROR: "execution policy" when activating venv
#   -> Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# ERROR: torch.cuda.is_available() returns False
#   -> Wrong PyTorch version installed. Uninstall and reinstall:
#   pip uninstall torch
#   pip install torch --index-url https://download.pytorch.org/whl/cu124

# ERROR: "Module not found"
#   -> Make sure venv is activated: .venv\Scripts\Activate.ps1
#   -> Make sure you're in the atlas/ directory

# GPU TEMPERATURE too high (above 85°C)
#   -> Reduce batch size
#   -> Improve case airflow / clean dust filters
#   -> Set a more aggressive fan curve in MSI Afterburner

# HOW TO MONITOR training
#   -> Open another PowerShell window and run:
#   nvidia-smi -l 5
#   -> This refreshes GPU stats every 5 seconds

# Vast.ai Guide for Atlas — Step by Step

Everything you need to do, in order.

---

## Step 1: Rent an Instance

1. Go to https://cloud.vast.ai/ and create an account
2. Add credit ($10–20 is enough to start, RTX 5090 costs ~$0.50–0.80/hr)
3. Go to **Templates** tab → search for `pytorch` → pick `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
4. Go to **Search** tab (GPU marketplace)
5. Filter by: **RTX 5090**, at least **50 GB disk**, the template you just selected
6. Click **RENT** on a machine with a good $/hr rate
7. Wait ~1 min for status to change to **Running**

## Step 2: Connect via SSH

1. In the Vast.ai console, click your running instance → **Connect**
2. Copy the SSH command — it looks like: `ssh -p 12345 root@123.45.67.89`
3. Run it in your terminal

## Step 3: Clone the Project

On the Vast.ai instance:

```bash
cd /workspace
git clone https://github.com/SH1NTSU/Atlas.git atlas
cd atlas
```

If the repo is private:
```bash
git clone https://SH1NTSU:YOUR_GITHUB_TOKEN@github.com/SH1NTSU/Atlas.git atlas
```

## Step 4: Run Setup

```bash
bash scripts/vastai_setup.sh
```

This installs Python, PyTorch+CUDA, and all dependencies. Takes ~3 min.
At the end it prints your GPU info — verify it says RTX 5090 and CUDA is available.

## Step 5: Start tmux

**IMPORTANT** — always run training inside tmux. If your SSH connection drops without tmux, training dies and you lose progress (while still paying).

```bash
tmux new -s train
```

Now everything you run is safe. If SSH drops, reconnect and run `tmux attach -t train`.

## Step 6: Download Training Data

```bash
source .venv/bin/activate
python scripts/download_data.py --output data/raw/ --max-size-gb 20
```

This streams from The Stack v2 and filters for quality. Takes **1–2 hours** for 20GB.
You'll see a progress bar per language.

## Step 7: Train Tokenizer

```bash
python src/tokenizer/trainer.py --data-dir data/raw/ --output data/tokenizer.json
```

Takes ~5 minutes. Creates `data/tokenizer.json` (32K vocab BPE tokenizer).

## Step 8: Tokenize Data into Binary

```bash
python src/data/preprocessing.py \
    --data-dir data/raw/ \
    --tokenizer data/tokenizer.json \
    --output data/processed/train.pt
```

Converts all JSONL files into a single binary tensor. Takes ~10 min.
At the end it prints the total token count — should be in the billions range for 20GB of code.

## Step 9: Start Training

```bash
python scripts/train.py \
    --phase pretrain \
    --precision bf16 \
    --batch-size 32 \
    --grad-accum 2 \
    --seq-length 2048 \
    --save-interval 5000 \
    --wandb
```

What the flags mean:
- `--precision bf16` — half precision, faster and uses less VRAM than fp32
- `--batch-size 32` — fits in 32GB VRAM of 5090
- `--grad-accum 2` — effective batch = 32 x 2 = 64
- `--save-interval 5000` — saves a checkpoint every 5000 steps (safety net)
- `--wandb` — optional, logs loss curves to wandb.ai (will ask you to login first time)

Training runs 100K steps. With a 5090, expect ~10K tokens/sec.

**Detach from tmux** while training runs: press `Ctrl+B`, then `D`.
You can close the SSH window. Training continues on the instance.

To check on it later: SSH back in, then `tmux attach -t train`.

## Step 10: Download Checkpoints to Your Machine

When training is done (or you want to grab an intermediate checkpoint), run this **from your local machine** (not the instance):

```bash
# Replace PORT and IP with your instance details from Vast.ai console
scp -P PORT -r root@IP_ADDRESS:/workspace/atlas/checkpoints/ ./checkpoints/
```

For faster transfer, compress first. On the **Vast.ai instance**:
```bash
cd /workspace/atlas
tar czf checkpoints.tar.gz checkpoints/
```

Then on **your machine**:
```bash
scp -P PORT root@IP_ADDRESS:/workspace/atlas/checkpoints.tar.gz .
tar xzf checkpoints.tar.gz
```

Also grab the tokenizer (you need it for inference):
```bash
scp -P PORT root@IP_ADDRESS:/workspace/atlas/data/tokenizer.json ./data/
```

## Step 11: Stop Paying

**After you've downloaded everything:**

1. Go to Vast.ai console → your instance
2. Click **Destroy** (the trash icon)
3. Confirm

This stops all billing immediately. The instance and all its data are deleted.

**If you want to pause and come back later:**
- Click **Stop** instead of Destroy — keeps your disk but stops compute billing
- You still pay a small storage fee (~$0.01–0.03/hr per GB)
- Click **Start** to resume later

**WARNING:** Do NOT just close the browser. The instance keeps running and billing until you explicitly stop/destroy it.

---

## Quick Copy-Paste Cheatsheet

Paste this entire block after SSH-ing into a fresh instance:

```bash
cd /workspace
git clone https://github.com/SH1NTSU/Atlas.git atlas
cd atlas
bash scripts/vastai_setup.sh
tmux new -s train
source .venv/bin/activate
python scripts/download_data.py --output data/raw/ --max-size-gb 20
python src/tokenizer/trainer.py --data-dir data/raw/ --output data/tokenizer.json
python src/data/preprocessing.py --data-dir data/raw/ --tokenizer data/tokenizer.json --output data/processed/train.pt
python scripts/train.py --phase pretrain --precision bf16 --batch-size 32 --grad-accum 2 --seq-length 2048 --save-interval 5000
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "CUDA not available" | You picked a wrong template. Destroy instance, re-rent with a PyTorch+CUDA template |
| Out of VRAM | Reduce `--batch-size` to 16 or 8 |
| SSH connection dropped | Reconnect, run `tmux attach -t train` — training should still be running |
| scp is slow | Use the tar+compress method from Step 10 |
| "Permission denied" on git clone | Use HTTPS with a GitHub token (see Step 3) |
| Training loss is NaN | Try `--precision fp32` instead of bf16, or reduce `--lr 1e-4` |

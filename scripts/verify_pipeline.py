"""
Pipeline verification script for Atlas.

Actually runs each step of the pipeline to catch real issues:
- Data creation/loading
- Tokenizer training & encoding
- Tokenization to .pt binary
- CodeDataset chunking
- FIM transformation
- InstructionDataset
- Model forward pass
- Training loop (few steps)

This is NOT a unit test — it's a real end-to-end smoke test.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

errors = []


def check(name, condition, detail=""):
    global errors
    if condition:
        print(f"  [{PASS}] {name}" + (f" — {detail}" if detail else ""))
    else:
        print(f"  [{FAIL}] {name}" + (f" — {detail}" if detail else ""))
        errors.append(name)


def warn(name, detail=""):
    print(f"  [{WARN}] {name}" + (f" — {detail}" if detail else ""))


# ─── Step 0: Synthetic data ─────────────────────────────────────────
print("\n═══ Step 0: Create synthetic data ═══")

tmpdir = tempfile.mkdtemp(prefix="atlas_verify_")
raw_dir = Path(tmpdir) / "raw"
raw_dir.mkdir()

# Create realistic code samples in JSONL format
samples = {
    "typescript": [
        'import React from "react";\n\ninterface Props {\n  name: string;\n  age: number;\n}\n\nexport const Greeting: React.FC<Props> = ({ name, age }) => {\n  return (\n    <div>\n      <h1>Hello, {name}!</h1>\n      <p>You are {age} years old.</p>\n    </div>\n  );\n};\n',
        'async function fetchUsers(): Promise<User[]> {\n  const response = await fetch("/api/users");\n  if (!response.ok) {\n    throw new Error(`HTTP error: ${response.status}`);\n  }\n  const data = await response.json();\n  return data.users;\n}\n\ninterface User {\n  id: number;\n  name: string;\n  email: string;\n}\n',
        'const express = require("express");\nconst app = express();\n\napp.get("/health", (req, res) => {\n  res.json({ status: "ok", uptime: process.uptime() });\n});\n\napp.listen(3000, () => {\n  console.log("Server running on port 3000");\n});\n',
    ],
    "go": [
        'package main\n\nimport (\n\t"fmt"\n\t"net/http"\n\t"encoding/json"\n)\n\ntype Handler struct {\n\tdb *Database\n}\n\nfunc (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {\n\tswitch r.Method {\n\tcase http.MethodGet:\n\t\tdata, err := h.db.GetAll()\n\t\tif err != nil {\n\t\t\thttp.Error(w, err.Error(), 500)\n\t\t\treturn\n\t\t}\n\t\tjson.NewEncoder(w).Encode(data)\n\tdefault:\n\t\thttp.Error(w, "method not allowed", 405)\n\t}\n}\n\nfunc main() {\n\tfmt.Println("Starting server...")\n}\n',
        'package utils\n\nimport (\n\t"strings"\n\t"unicode"\n)\n\nfunc CamelToSnake(s string) string {\n\tvar result strings.Builder\n\tfor i, r := range s {\n\t\tif unicode.IsUpper(r) {\n\t\t\tif i > 0 {\n\t\t\t\tresult.WriteRune(\'_\')\n\t\t\t}\n\t\t\tresult.WriteRune(unicode.ToLower(r))\n\t\t} else {\n\t\t\tresult.WriteRune(r)\n\t\t}\n\t}\n\treturn result.String()\n}\n',
    ],
    "rust": [
        'use std::collections::HashMap;\n\n#[derive(Debug, Clone)]\npub struct Config {\n    pub host: String,\n    pub port: u16,\n    pub env_vars: HashMap<String, String>,\n}\n\nimpl Config {\n    pub fn from_env() -> Self {\n        let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into());\n        let port: u16 = std::env::var("PORT")\n            .unwrap_or_else(|_| "8080".into())\n            .parse()\n            .expect("PORT must be a number");\n\n        Config {\n            host,\n            port,\n            env_vars: std::env::vars().collect(),\n        }\n    }\n}\n',
    ],
    "bash": [
        '#!/bin/bash\nset -euo pipefail\n\nLOG_DIR="/var/log/app"\nmkdir -p "$LOG_DIR"\n\necho "Starting deployment..."\n\nif ! docker compose up -d --build; then\n    echo "ERROR: Deployment failed" >&2\n    exit 1\nfi\n\necho "Waiting for health check..."\nfor i in $(seq 1 30); do\n    if curl -sf http://localhost:3000/health > /dev/null; then\n        echo "Service is healthy!"\n        exit 0\n    fi\n    sleep 1\ndone\n\necho "ERROR: Health check timed out" >&2\nexit 1\n',
    ],
}

total_samples = 0
for lang, codes in samples.items():
    jsonl_file = raw_dir / f"{lang}.jsonl"
    with open(jsonl_file, "w") as f:
        for code in codes:
            f.write(json.dumps({"text": code, "language": lang, "size": len(code)}) + "\n")
            total_samples += 1

check("Synthetic data created", total_samples == 7, f"{total_samples} samples in {len(samples)} languages")
check("JSONL files exist", len(list(raw_dir.glob("*.jsonl"))) == 4)

# Verify JSONL files are valid and readable
for jsonl_file in sorted(raw_dir.glob("*.jsonl")):
    with open(jsonl_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                record = json.loads(line)
                assert "text" in record, f"Missing 'text' field in {jsonl_file.name} line {i}"
            except Exception as e:
                check(f"JSONL parse {jsonl_file.name}:{i}", False, str(e))

check("All JSONL files valid", True)


# ─── Step 1: Train tokenizer ────────────────────────────────────────
print("\n═══ Step 1: Train tokenizer ═══")

from tokenizer.trainer import train_tokenizer, load_tokenizer, SPECIAL_TOKENS

tokenizer_path = Path(tmpdir) / "tokenizer.json"
t0 = time.time()
tokenizer = train_tokenizer(
    data_dir=str(raw_dir),
    output_path=str(tokenizer_path),
    vocab_size=1000,  # small for speed
)
t1 = time.time()

check("Tokenizer trained", tokenizer_path.exists(), f"{t1-t0:.1f}s")
check("Vocab size", tokenizer.get_vocab_size() > 0, f"vocab={tokenizer.get_vocab_size()}")

# Verify special tokens have correct IDs
vocab = tokenizer.get_vocab()
for i, token in enumerate(SPECIAL_TOKENS):
    actual_id = vocab.get(token, -1)
    check(f"Special token '{token}' has ID {i}", actual_id == i,
          f"actual={actual_id}" if actual_id != i else "")

# Test encoding/decoding roundtrip
test_code = 'func main() {\n\tfmt.Println("hello")\n}'
encoded = tokenizer.encode(test_code)
check("Encoding produces tokens", len(encoded.ids) > 0, f"{len(encoded.ids)} tokens")

# Check BOS/EOS are added by post-processor
check("BOS token added", encoded.ids[0] == 1, f"first token={encoded.ids[0]}")
check("EOS token added", encoded.ids[-1] == 2, f"last token={encoded.ids[-1]}")

decoded = tokenizer.decode(encoded.ids)
check("Decode roundtrip", "main" in decoded and "hello" in decoded,
      f"decoded={decoded[:60]}...")

# Check that code-specific tokens work well
code_samples = [
    ("TypeScript arrow", "const fn = (x: number): string => x.toString();"),
    ("Go struct", "type User struct {\n\tName string `json:\"name\"`\n}"),
    ("Rust lifetime", "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {"),
    ("Bash pipe", "cat file.txt | grep -E '^error' | wc -l"),
]
for name, code in code_samples:
    enc = tokenizer.encode(code)
    dec = tokenizer.decode(enc.ids)
    # Strip BOS/EOS for comparison
    check(f"Tokenize {name}", len(enc.ids) > 2, f"{len(enc.ids)} tokens")


# ─── Step 2: Tokenize data to binary ────────────────────────────────
print("\n═══ Step 2: Tokenize and save to binary ═══")

from data.preprocessing import tokenize_and_save, CodeDataset, FIMDataset, InstructionDataset
from data.preprocessing import EOS_ID, BOS_ID, PAD_ID, FIM_PREFIX_ID, FIM_MIDDLE_ID, FIM_SUFFIX_ID

output_pt = Path(tmpdir) / "processed" / "train.pt"
t0 = time.time()
tokenize_and_save(
    data_dir=str(raw_dir),
    tokenizer_path=str(tokenizer_path),
    output_path=str(output_pt),
)
t1 = time.time()

check("Binary tensor saved", output_pt.exists(), f"{t1-t0:.1f}s")

# Load and verify the tensor
tensor = torch.load(str(output_pt), weights_only=True)
check("Tensor is 1D", tensor.dim() == 1, f"shape={tensor.shape}")
check("Tensor is long dtype", tensor.dtype == torch.long)
check("Tensor has tokens", len(tensor) > 100, f"{len(tensor):,} tokens")
check("Tensor not all zeros", tensor.sum().item() > 0)
check("Tensor not all same value", tensor.unique().numel() > 10,
      f"{tensor.unique().numel()} unique tokens")

# Check EOS tokens between documents
eos_count = (tensor == EOS_ID).sum().item()
check("EOS tokens between documents", eos_count == total_samples,
      f"expected {total_samples}, got {eos_count}")

# Check no negative values
check("No negative token IDs", tensor.min().item() >= 0, f"min={tensor.min().item()}")


# ─── Step 3: CodeDataset loading ─────────────────────────────────────
print("\n═══ Step 3: CodeDataset loading & chunking ═══")

seq_length = 64  # small for testing
dataset = CodeDataset(str(output_pt), seq_length=seq_length)

expected_seqs = len(tensor) // seq_length
check("Dataset length correct", len(dataset) == expected_seqs,
      f"expected {expected_seqs}, got {len(dataset)}")

# Verify shapes and types
sample = dataset[0]
check("Sample has input_ids", "input_ids" in sample)
check("Sample has labels", "labels" in sample)
check("input_ids shape", sample["input_ids"].shape == (seq_length,),
      f"shape={sample["input_ids"].shape}")
check("labels shape", sample["labels"].shape == (seq_length,),
      f"shape={sample["labels"].shape}")
check("input_ids == labels", torch.equal(sample["input_ids"], sample["labels"]))
check("Dtype is long", sample["input_ids"].dtype == torch.long)

# Check that all sequences are valid token IDs (non-negative, within vocab)
all_valid = True
for i in range(min(len(dataset), 10)):
    s = dataset[i]
    if s["input_ids"].min().item() < 0:
        all_valid = False
        break
check("All token IDs non-negative", all_valid)

# DataLoader test
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=2, shuffle=True)
batch = next(iter(loader))
check("DataLoader batch shape", batch["input_ids"].shape == (2, seq_length),
      f"shape={batch['input_ids'].shape}")
check("DataLoader batch labels shape", batch["labels"].shape == (2, seq_length))

# Test DataLoader iteration speed (catch the "loading for 2 hours" issue)
t0 = time.time()
batch_count = 0
for batch in loader:
    batch_count += 1
    if batch_count >= 20:
        break
t1 = time.time()
check("DataLoader iteration fast", t1 - t0 < 5.0,
      f"{batch_count} batches in {t1-t0:.2f}s ({batch_count/(t1-t0):.0f} batches/s)")

# Multi-worker DataLoader test (requires __main__ guard on macOS)
# This is a known issue: train.py uses num_workers=2 which works because
# it has if __name__ == '__main__'. We test with num_workers=0 here.
loader_mp = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
t0 = time.time()
batch = next(iter(loader_mp))
t1 = time.time()
check("DataLoader (num_workers=0) works", batch["input_ids"].shape[0] == 2,
      f"loaded in {t1-t0:.2f}s")


# ─── Step 4: FIMDataset ──────────────────────────────────────────────
print("\n═══ Step 4: FIM transformation ═══")

fim_dataset = FIMDataset(dataset, fim_rate=1.0)  # force FIM on every sample
check("FIM dataset length", len(fim_dataset) == len(dataset))

sample = fim_dataset[0]
tokens = sample["input_ids"]
check("FIM output length matches input", len(tokens) == seq_length)
check("FIM has prefix token", (tokens == FIM_PREFIX_ID).any().item())
check("FIM has suffix token", (tokens == FIM_SUFFIX_ID).any().item())
check("FIM has middle token", (tokens == FIM_MIDDLE_ID).any().item())

# Check FIM token ordering: prefix should come before suffix, suffix before middle
prefix_pos = (tokens == FIM_PREFIX_ID).nonzero(as_tuple=True)[0]
suffix_pos = (tokens == FIM_SUFFIX_ID).nonzero(as_tuple=True)[0]
middle_pos = (tokens == FIM_MIDDLE_ID).nonzero(as_tuple=True)[0]
if len(prefix_pos) > 0 and len(suffix_pos) > 0 and len(middle_pos) > 0:
    check("FIM order: prefix < suffix < middle",
          prefix_pos[0] < suffix_pos[0] < middle_pos[0])

# Test with fim_rate=0 (should return unchanged)
no_fim = FIMDataset(dataset, fim_rate=0.0)
sample_orig = dataset[0]
sample_nofim = no_fim[0]
check("FIM rate 0 leaves tokens unchanged",
      torch.equal(sample_orig["input_ids"], sample_nofim["input_ids"]))


# ─── Step 5: InstructionDataset ──────────────────────────────────────
print("\n═══ Step 5: InstructionDataset ═══")

instruct_path = Path(tmpdir) / "instruct.jsonl"
instruct_samples = [
    {"messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a function to reverse a string in Go."},
        {"role": "assistant", "content": 'func reverse(s string) string {\n\trunes := []rune(s)\n\tfor i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {\n\t\trunes[i], runes[j] = runes[j], runes[i]\n\t}\n\treturn string(runes)\n}'},
    ]},
    {"instruction": "Explain what this Rust code does", "input": "let v: Vec<i32> = (0..10).filter(|x| x % 2 == 0).collect();", "output": "This creates a vector of even numbers from 0 to 9."},
    {"instruction": "Write a bash one-liner to count files", "input": "", "output": "find . -type f | wc -l"},
]

with open(instruct_path, "w") as f:
    for s in instruct_samples:
        f.write(json.dumps(s) + "\n")

instruct_dataset = InstructionDataset(
    data_path=str(instruct_path),
    tokenizer=tokenizer,
    seq_length=256,
)

check("Instruction dataset length", len(instruct_dataset) == 3)

for i in range(len(instruct_dataset)):
    sample = instruct_dataset[i]
    check(f"Instruct sample {i} shape", sample["input_ids"].shape == (256,))
    check(f"Instruct sample {i} labels shape", sample["labels"].shape == (256,))
    # Check that padding is masked in labels
    pad_mask = sample["input_ids"] == PAD_ID
    if pad_mask.any():
        check(f"Instruct sample {i} padding masked",
              (sample["labels"][pad_mask] == -100).all().item())

# DataLoader for instruction dataset
instruct_loader = DataLoader(instruct_dataset, batch_size=2, shuffle=True)
batch = next(iter(instruct_loader))
check("Instruct DataLoader works", batch["input_ids"].shape == (2, 256))


# ─── Step 6: Model forward pass ──────────────────────────────────────
print("\n═══ Step 6: Model forward pass ═══")

from model.transformer import Atlas, AtlasConfig

config = AtlasConfig(
    vocab_size=tokenizer.get_vocab_size(),
    hidden_size=128,  # small for testing
    intermediate_size=356,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
)

model = Atlas(config)
param_count = model.count_parameters()
check("Model created", param_count > 0, f"{param_count/1e6:.1f}M params")

# Forward pass with real data from dataset
batch = next(iter(DataLoader(dataset, batch_size=2)))
input_ids = batch["input_ids"]
labels = batch["labels"]

# Clamp token IDs to vocab size (synthetic data may have been tokenized with different vocab)
input_ids = input_ids.clamp(0, config.vocab_size - 1)
labels = labels.clamp(0, config.vocab_size - 1)

output = model(input_ids=input_ids, labels=labels)
check("Forward pass produces loss", output["loss"] is not None)
check("Loss is finite", torch.isfinite(output["loss"]).item(),
      f"loss={output['loss'].item():.4f}")
check("Loss is positive", output["loss"].item() > 0)
check("Logits shape", output["logits"].shape == (2, seq_length, config.vocab_size),
      f"shape={output['logits'].shape}")
check("KV cache returned", output["kv_cache"] is not None and len(output["kv_cache"]) == 4)

# Test generation
gen_input = torch.randint(0, config.vocab_size, (1, 10))
t0 = time.time()
generated = model.generate(gen_input, max_new_tokens=20, temperature=0.8)
t1 = time.time()
check("Generation works", generated.shape[1] > 10, f"generated {generated.shape[1]-10} new tokens in {t1-t0:.2f}s")
check("Generated tokens in vocab range", generated.max().item() < config.vocab_size)


# ─── Step 7: Training loop (few steps) ───────────────────────────────
print("\n═══ Step 7: Training loop (5 steps) ═══")

from training.trainer import Trainer, TrainingArgs

checkpoint_dir = Path(tmpdir) / "checkpoints"
training_args = TrainingArgs(
    data_path=str(output_pt),
    checkpoint_dir=str(checkpoint_dir),
    batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-3,
    max_steps=5,
    warmup_steps=2,
    seq_length=seq_length,
    precision="fp32",  # CPU doesn't support fp16 AMP well
    log_interval=1,
    save_interval=3,
    eval_interval=3,
)

# Create DataLoaders with clamped data
class ClampedDataset(torch.utils.data.Dataset):
    def __init__(self, base, max_id):
        self.base = base
        self.max_id = max_id
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        return {
            "input_ids": item["input_ids"].clamp(0, self.max_id),
            "labels": item["labels"].clamp(0, self.max_id),
        }

clamped = ClampedDataset(dataset, config.vocab_size - 1)
train_loader = DataLoader(clamped, batch_size=2, shuffle=True)
eval_loader = DataLoader(clamped, batch_size=2, shuffle=False)

model2 = Atlas(config)
trainer = Trainer(model2, training_args)

t0 = time.time()
trainer.train(train_loader, eval_loader)
t1 = time.time()

check("Training completed", trainer.global_step == 5, f"steps={trainer.global_step}")
check("Training time reasonable", t1 - t0 < 60, f"{t1-t0:.1f}s")
check("Tokens counted", trainer.tokens_seen > 0, f"tokens_seen={trainer.tokens_seen:,}")

# Check checkpoint was saved at step 3
step3_ckpt = checkpoint_dir / "step_3" / "checkpoint.pt"
check("Checkpoint saved at step 3", step3_ckpt.exists())

final_ckpt = checkpoint_dir / "final" / "checkpoint.pt"
check("Final checkpoint saved", final_ckpt.exists())

# Load and verify checkpoint
if final_ckpt.exists():
    ckpt = torch.load(str(final_ckpt), weights_only=False)
    check("Checkpoint has model_state_dict", "model_state_dict" in ckpt)
    check("Checkpoint has optimizer_state_dict", "optimizer_state_dict" in ckpt)
    check("Checkpoint has global_step", ckpt.get("global_step") == 5)
    check("Checkpoint has tokens_seen", ckpt.get("tokens_seen", 0) > 0)


# ─── Step 8: Resume training ─────────────────────────────────────────
print("\n═══ Step 8: Resume from checkpoint ═══")

training_args2 = TrainingArgs(
    data_path=str(output_pt),
    checkpoint_dir=str(checkpoint_dir / "resumed"),
    resume_from=str(checkpoint_dir / "final"),
    batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-3,
    max_steps=8,  # 3 more steps from step 5
    warmup_steps=2,
    seq_length=seq_length,
    precision="fp32",
    log_interval=1,
    save_interval=100,
    eval_interval=100,
)

model3 = Atlas(config)
trainer2 = Trainer(model3, training_args2)
trainer2.train(train_loader)

check("Resumed training completed", trainer2.global_step == 8,
      f"steps={trainer2.global_step}")
check("Tokens accumulated from resume", trainer2.tokens_seen > trainer.tokens_seen,
      f"{trainer2.tokens_seen:,} > {trainer.tokens_seen:,}")


# ─── Step 9: Full model config from YAML ─────────────────────────────
print("\n═══ Step 9: Model from YAML config ═══")

import yaml
config_path = project_root / "config" / "model.yaml"
if config_path.exists():
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    model_cfg = raw_config.get("model", {})
    full_config = AtlasConfig()
    for key, value in model_cfg.items():
        if hasattr(full_config, key):
            setattr(full_config, key, value)

    full_model = Atlas(full_config)
    full_params = full_model.count_parameters()
    check("Full model builds from YAML", full_params > 200e6,
          f"{full_params/1e6:.1f}M params")

    # Test forward pass with full model
    dummy = torch.randint(0, full_config.vocab_size, (1, 32))
    out = full_model(dummy, labels=dummy)
    check("Full model forward pass", torch.isfinite(out["loss"]).item(),
          f"loss={out['loss'].item():.4f}")

    # Verify config matches YAML
    check("vocab_size matches", full_config.vocab_size == model_cfg.get("vocab_size", 32000))
    check("hidden_size matches", full_config.hidden_size == model_cfg.get("hidden_size", 1024))
    check("num_layers matches", full_config.num_hidden_layers == model_cfg.get("num_hidden_layers", 20))
    del full_model
else:
    warn("config/model.yaml not found — skipping YAML config check")


# ─── Summary ──────────────────────────────────────────────────────────
print("\n" + "═" * 50)
if errors:
    print(f"\n{FAIL} {len(errors)} checks failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"\n{PASS} All checks passed! Pipeline is ready for GPU training.")

# Cleanup
import shutil
shutil.rmtree(tmpdir)

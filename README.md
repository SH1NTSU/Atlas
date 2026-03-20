# Atlas — Personal AI Coding Assistant

A ~150M parameter transformer model trained from scratch for:
- Code completion & generation (TSX, Go, Rust, Bash)
- Code explanation & Q&A
- Shell commands, Docker, DevOps configs
- Research summarization

## Architecture
- Llama-style transformer (RoPE, RMSNorm, SwiGLU, GQA)
- ~150M parameters
- 2048 token context window
- Custom BPE tokenizer optimized for target languages

## Hardware Target
- RTX 3060 Ti 8GB VRAM
- Training in fp16 mixed precision

## Project Structure
```
atlas/
├── config/              # Model and training configs
├── src/
│   ├── model/           # Model architecture (from scratch)
│   ├── tokenizer/       # Custom BPE tokenizer
│   ├── data/            # Data pipeline & filtering
│   ├── training/        # Training loops
│   └── inference/       # Serving & generation
├── scripts/             # Data download, preprocessing, training launch
├── data/                # Raw and processed datasets (gitignored)
├── checkpoints/         # Model checkpoints (gitignored)
└── tests/               # Unit tests
```

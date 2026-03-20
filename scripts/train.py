"""
Main training entry point for Atlas.

Usage:
    # Pretrain from scratch
    python scripts/train.py --phase pretrain

    # Resume training
    python scripts/train.py --phase pretrain --resume checkpoints/step_5000

    # Instruction fine-tune (after pretraining)
    python scripts/train.py --phase instruct --resume checkpoints/final
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from torch.utils.data import DataLoader, random_split

from model.transformer import Atlas, AtlasConfig
from data.preprocessing import CodeDataset, FIMDataset, InstructionDataset
from training.trainer import Trainer, TrainingArgs


def build_model(config_path: str = None) -> tuple[Atlas, AtlasConfig]:
    """Build model from config."""
    config = AtlasConfig()

    if config_path:
        import yaml
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        model_cfg = raw.get("model", {})
        for key, value in model_cfg.items():
            if hasattr(config, key):
                setattr(config, key, value)

    model = Atlas(config)
    param_count = model.count_parameters()
    print(f"Atlas model: {param_count / 1e6:.1f}M parameters")
    return model, config


def train_pretrain(args):
    """Phase 1: Pretrain on raw code."""
    model, config = build_model(args.config)

    # Load dataset
    print(f"Loading data from {args.data}...")
    base_dataset = CodeDataset(args.data, seq_length=args.seq_length)
    print(f"Dataset: {len(base_dataset):,} sequences of {args.seq_length} tokens")

    # Apply FIM transformation
    dataset = FIMDataset(base_dataset, fim_rate=args.fim_rate)

    # Train/eval split (95/5)
    train_size = int(0.95 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # Training args
    training_args = TrainingArgs(
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        seq_length=args.seq_length,
        precision=args.precision,
        fim_rate=args.fim_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        use_wandb=args.wandb,
    )

    trainer = Trainer(model, training_args)
    trainer.train(train_loader, eval_loader)


def train_instruct(args):
    """Phase 2: Instruction fine-tuning with LoRA."""
    model, config = build_model(args.config)

    # Load pretrained weights
    if args.resume:
        checkpoint = torch.load(
            Path(args.resume) / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded pretrained weights from {args.resume}")

    # Load instruction dataset
    from tokenizer.trainer import load_tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    dataset = InstructionDataset(
        data_path=args.instruct_data,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
    )
    print(f"Instruction dataset: {len(dataset):,} samples")

    # Train/eval split
    train_size = int(0.95 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    training_args = TrainingArgs(
        data_path=args.instruct_data,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-5,  # Lower LR for fine-tuning
        max_steps=args.max_steps,
        warmup_steps=200,
        seq_length=args.seq_length,
        precision=args.precision,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        use_wandb=args.wandb,
    )

    trainer = Trainer(model, training_args)
    trainer.train(train_loader, eval_loader)


def main():
    parser = argparse.ArgumentParser(description="Train Atlas model")
    parser.add_argument("--phase", choices=["pretrain", "instruct"], required=True)
    parser.add_argument("--config", default="config/model.yaml", help="Model config file")
    parser.add_argument("--data", default="data/processed/train.pt", help="Pretrain data path")
    parser.add_argument("--instruct-data", default="data/instruct/train.jsonl", help="Instruction data path")
    parser.add_argument("--tokenizer", default="data/tokenizer.json", help="Tokenizer path")
    parser.add_argument("--checkpoint-dir", default="checkpoints/", help="Checkpoint directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--fim-rate", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    if args.phase == "pretrain":
        train_pretrain(args)
    elif args.phase == "instruct":
        train_instruct(args)


if __name__ == "__main__":
    main()

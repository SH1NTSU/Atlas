"""
Atlas training loop.

Supports:
- Mixed precision (fp16) training
- Gradient accumulation
- Gradient clipping
- Cosine learning rate schedule with warmup
- Checkpointing and resume
- W&B logging (optional)
- Works on both Linux and Windows
"""

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import yaml
from tqdm import tqdm


@dataclass
class TrainingArgs:
    # Paths
    config_path: str = "config/model.yaml"
    data_path: str = "data/processed/train.pt"
    tokenizer_path: str = "data/tokenizer.json"
    checkpoint_dir: str = "checkpoints/"
    resume_from: Optional[str] = None

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    max_steps: int = 100000
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"
    seq_length: int = 2048

    # Precision
    precision: str = "fp16"  # "fp16", "bf16", or "fp32"

    # FIM
    fim_rate: float = 0.5

    # Logging
    log_interval: int = 10
    save_interval: int = 5000
    eval_interval: int = 1000
    use_wandb: bool = False
    wandb_project: str = "atlas"


def get_lr(step: int, args: TrainingArgs) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < args.warmup_steps:
        return args.learning_rate * step / args.warmup_steps

    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine_decay


class Trainer:
    def __init__(self, model: nn.Module, args: TrainingArgs):
        self.model = model
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)

        # Optimizer — AdamW with weight decay only on non-bias, non-norm params
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "norm" in name or "bias" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=args.learning_rate, betas=(0.9, 0.95))

        # Mixed precision
        self.use_amp = args.precision in ("fp16", "bf16") and torch.cuda.is_available()
        self.amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
        self.amp_device = "cuda" if torch.cuda.is_available() else "cpu"
        scaler_enabled = args.precision == "fp16" and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(self.amp_device, enabled=scaler_enabled)

        # Tracking
        self.global_step = 0
        self.tokens_seen = 0
        self.best_loss = float("inf")

        # Checkpoint dir
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Main training loop."""
        args = self.args
        model = self.model
        model.train()

        # W&B init
        if args.use_wandb:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))

        # Resume from checkpoint
        if args.resume_from:
            self._load_checkpoint(args.resume_from)

        print(f"Training on {self.device}")
        print(f"Precision: {args.precision}")
        print(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps} = "
              f"{args.batch_size * args.gradient_accumulation_steps} effective")
        print(f"Max steps: {args.max_steps}")
        print()

        accumulation_loss = 0.0
        start_time = time.time()
        data_iter = iter(train_dataloader)

        pbar = tqdm(range(self.global_step, args.max_steps), desc="Training", initial=self.global_step)

        for step in pbar:
            # Update learning rate
            lr = get_lr(step, args)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient accumulation loop
            self.optimizer.zero_grad()
            for micro_step in range(args.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast(self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"] / args.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accumulation_loss += loss.item()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step = step + 1
            self.tokens_seen += args.batch_size * args.gradient_accumulation_steps * args.seq_length

            # Logging
            if self.global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = self.tokens_seen / elapsed
                pbar.set_postfix({
                    "loss": f"{accumulation_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{tokens_per_sec:.0f}",
                    "grad": f"{grad_norm:.2f}",
                })

                if args.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": accumulation_loss,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/tokens_seen": self.tokens_seen,
                    }, step=self.global_step)

            accumulation_loss = 0.0

            # Evaluation
            if eval_dataloader and self.global_step % args.eval_interval == 0:
                eval_loss = self.evaluate(eval_dataloader)
                print(f"\nStep {self.global_step}: eval_loss={eval_loss:.4f}")
                model.train()

                if args.use_wandb:
                    import wandb
                    wandb.log({"eval/loss": eval_loss}, step=self.global_step)

            # Checkpointing
            if self.global_step % args.save_interval == 0:
                self._save_checkpoint()

        # Save final checkpoint
        self._save_checkpoint(final=True)
        print(f"\nTraining complete. Total tokens: {self.tokens_seen:,}")

    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader, max_batches: int = 50) -> float:
        """Evaluate model on eval set."""
        self.model.eval()
        total_loss = 0.0
        count = 0

        for batch in eval_dataloader:
            if count >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast(self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(input_ids=input_ids, labels=labels)

            total_loss += outputs["loss"].item()
            count += 1

        return total_loss / max(count, 1)

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        tag = "final" if final else f"step_{self.global_step}"
        path = Path(self.args.checkpoint_dir) / tag

        from dataclasses import asdict
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": asdict(self.model.config) if hasattr(self.model, "config") else {},
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_loss": self.best_loss,
            "args": vars(self.args),
        }
        path.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path / "checkpoint.pt")
        print(f"\nCheckpoint saved: {path}")

    def _load_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        checkpoint_path = Path(path) / "checkpoint.pt"
        print(f"Resuming from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.tokens_seen = checkpoint["tokens_seen"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Resumed at step {self.global_step}, tokens_seen={self.tokens_seen:,}")

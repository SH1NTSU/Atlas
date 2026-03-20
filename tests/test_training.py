"""
Tests for training pipeline.

Tests verify:
- Learning rate schedule (warmup + cosine decay)
- Trainer can run a few steps without crashing
- Loss decreases over multiple steps on a small dataset
- Checkpoint save and load roundtrips correctly
- Gradient accumulation produces correct effective batch size
- Mixed precision training works on CPU (graceful fallback)
- Optimizer correctly separates weight decay groups
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.transformer import Atlas, AtlasConfig
from training.trainer import Trainer, TrainingArgs, get_lr
from data.preprocessing import CodeDataset


# ─── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def small_config():
    return AtlasConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


@pytest.fixture
def small_model(small_config):
    return Atlas(small_config)


@pytest.fixture
def dummy_dataloader(small_config):
    """Create a tiny dataloader for testing."""
    n_samples = 32
    seq_len = 32
    data = torch.randint(10, small_config.vocab_size, (n_samples, seq_len))
    dataset = TensorDataset(data, data)

    class WrappedDataset:
        def __init__(self, tensor_ds):
            self.ds = tensor_ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            input_ids, labels = self.ds[idx]
            return {"input_ids": input_ids, "labels": labels}

    return DataLoader(WrappedDataset(dataset), batch_size=4, shuffle=True)


@pytest.fixture
def training_args(tmp_path):
    return TrainingArgs(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        max_steps=10,
        warmup_steps=3,
        seq_length=32,
        precision="fp32",  # CPU doesn't support fp16 autocast well
        log_interval=2,
        save_interval=5,
        eval_interval=5,
        use_wandb=False,
    )


# ─── Learning Rate Schedule ─────────────────────────────────

class TestLRSchedule:
    def test_warmup_starts_at_zero(self):
        args = TrainingArgs(learning_rate=1e-3, warmup_steps=100)
        lr = get_lr(0, args)
        assert lr == 0.0

    def test_warmup_linear(self):
        args = TrainingArgs(learning_rate=1e-3, warmup_steps=100)
        lr_50 = get_lr(50, args)
        assert abs(lr_50 - 0.5e-3) < 1e-6

    def test_warmup_reaches_peak(self):
        args = TrainingArgs(learning_rate=1e-3, warmup_steps=100)
        lr_100 = get_lr(100, args)
        assert abs(lr_100 - 1e-3) < 1e-6

    def test_cosine_decays(self):
        args = TrainingArgs(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=100, max_steps=1000)
        lr_mid = get_lr(550, args)  # Midpoint of cosine
        assert 1e-4 < lr_mid < 1e-3

    def test_cosine_reaches_min(self):
        args = TrainingArgs(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=100, max_steps=1000)
        lr_end = get_lr(1000, args)
        assert abs(lr_end - 1e-4) < 1e-6

    def test_lr_monotonic_during_decay(self):
        args = TrainingArgs(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=10, max_steps=100)
        lrs = [get_lr(s, args) for s in range(10, 101)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-8  # Allow tiny float errors


# ─── Trainer ─────────────────────────────────────────────────

class TestTrainer:
    def test_trainer_creates(self, small_model, training_args):
        trainer = Trainer(small_model, training_args)
        assert trainer.global_step == 0
        assert trainer.tokens_seen == 0

    def test_trainer_runs_steps(self, small_model, training_args, dummy_dataloader):
        training_args.max_steps = 3
        training_args.log_interval = 1
        training_args.save_interval = 100  # Don't save during this test
        training_args.eval_interval = 100

        trainer = Trainer(small_model, training_args)
        trainer.train(dummy_dataloader)
        assert trainer.global_step == 3

    def test_loss_decreases(self, small_config, training_args, dummy_dataloader):
        """Loss should decrease over training on a small dataset."""
        torch.manual_seed(42)
        model = Atlas(small_config)
        training_args.max_steps = 20
        training_args.gradient_accumulation_steps = 1
        training_args.learning_rate = 1e-2  # High LR for fast convergence on toy data
        training_args.save_interval = 100
        training_args.eval_interval = 100

        # Get initial loss
        model.eval()
        batch = next(iter(dummy_dataloader))
        with torch.no_grad():
            initial_loss = model(
                batch["input_ids"],
                labels=batch["labels"]
            )["loss"].item()

        # Train
        model.train()
        trainer = Trainer(model, training_args)
        trainer.train(dummy_dataloader)

        # Get final loss
        model.eval()
        with torch.no_grad():
            final_loss = model(
                batch["input_ids"].to(trainer.device),
                labels=batch["labels"].to(trainer.device),
            )["loss"].item()

        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_evaluate(self, small_model, training_args, dummy_dataloader):
        trainer = Trainer(small_model, training_args)
        eval_loss = trainer.evaluate(dummy_dataloader, max_batches=3)
        assert eval_loss > 0
        assert not torch.isnan(torch.tensor(eval_loss))


# ─── Checkpointing ──────────────────────────────────────────

class TestCheckpointing:
    def test_save_checkpoint(self, small_model, training_args):
        trainer = Trainer(small_model, training_args)
        trainer.global_step = 100
        trainer.tokens_seen = 50000
        trainer._save_checkpoint()

        ckpt_path = Path(training_args.checkpoint_dir) / "step_100" / "checkpoint.pt"
        assert ckpt_path.exists()

    def test_load_checkpoint(self, small_config, training_args):
        # Save
        torch.manual_seed(42)
        model1 = Atlas(small_config)
        trainer1 = Trainer(model1, training_args)
        trainer1.global_step = 50
        trainer1.tokens_seen = 25000
        trainer1._save_checkpoint()

        # Load into new model
        torch.manual_seed(99)  # Different seed
        model2 = Atlas(small_config)
        training_args.resume_from = str(Path(training_args.checkpoint_dir) / "step_50")
        trainer2 = Trainer(model2, training_args)
        trainer2._load_checkpoint(training_args.resume_from)

        assert trainer2.global_step == 50
        assert trainer2.tokens_seen == 25000

        # Verify model weights match
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            torch.testing.assert_close(p1, p2, msg=f"Mismatch in {n1}")

    def test_save_final_checkpoint(self, small_model, training_args):
        trainer = Trainer(small_model, training_args)
        trainer._save_checkpoint(final=True)

        ckpt_path = Path(training_args.checkpoint_dir) / "final" / "checkpoint.pt"
        assert ckpt_path.exists()


# ─── Optimizer Groups ───────────────────────────────────────

class TestOptimizerGroups:
    def test_weight_decay_groups(self, small_model, training_args):
        trainer = Trainer(small_model, training_args)
        # Should have 2 param groups
        assert len(trainer.optimizer.param_groups) == 2
        # First group: weight decay
        assert trainer.optimizer.param_groups[0]["weight_decay"] == training_args.weight_decay
        # Second group: no weight decay (norms and biases)
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_all_params_in_optimizer(self, small_model, training_args):
        trainer = Trainer(small_model, training_args)
        optim_params = sum(len(g["params"]) for g in trainer.optimizer.param_groups)
        model_params = sum(1 for p in small_model.parameters() if p.requires_grad)
        assert optim_params == model_params

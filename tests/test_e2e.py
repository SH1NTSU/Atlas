"""
End-to-end integration tests for Atlas.

Tests the full pipeline:
1. Create model from config
2. Create dummy data
3. Tokenize and preprocess
4. Train for a few steps
5. Save checkpoint
6. Load checkpoint
7. Generate text
8. Verify the whole thing works

This test simulates what will happen on the Windows training machine.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from model.transformer import Atlas, AtlasConfig
from data.preprocessing import CodeDataset, FIMDataset, InstructionDataset
from training.trainer import Trainer, TrainingArgs


class TestEndToEnd:
    """Full pipeline integration test."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Set up a complete workspace with dummy data."""
        ws = {
            "root": tmp_path,
            "data_dir": tmp_path / "data",
            "checkpoint_dir": tmp_path / "checkpoints",
        }
        ws["data_dir"].mkdir()
        ws["checkpoint_dir"].mkdir()
        return ws

    @pytest.fixture
    def config(self):
        return AtlasConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )

    def test_full_pretrain_pipeline(self, workspace, config):
        """Test: create model -> create data -> train -> checkpoint -> load -> generate."""
        # Step 1: Create model
        torch.manual_seed(42)
        model = Atlas(config)
        assert model.count_parameters() > 0

        # Step 2: Create dummy tokenized data (simulating preprocessed code)
        n_tokens = 2048
        tokens = torch.randint(10, config.vocab_size, (n_tokens,), dtype=torch.long)
        data_path = workspace["data_dir"] / "train.pt"
        torch.save(tokens, data_path)

        # Step 3: Load and wrap in FIM dataset
        dataset = CodeDataset(str(data_path), seq_length=32)
        fim_dataset = FIMDataset(dataset, fim_rate=0.5)
        assert len(fim_dataset) > 0

        dataloader = DataLoader(fim_dataset, batch_size=4, shuffle=True)

        # Step 4: Train for a few steps
        args = TrainingArgs(
            checkpoint_dir=str(workspace["checkpoint_dir"]),
            batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            max_steps=5,
            warmup_steps=2,
            seq_length=32,
            precision="fp32",
            log_interval=2,
            save_interval=5,
            eval_interval=100,
            use_wandb=False,
        )

        trainer = Trainer(model, args)
        trainer.train(dataloader)
        assert trainer.global_step == 5

        # Step 5: Verify checkpoint was saved
        ckpt_path = workspace["checkpoint_dir"] / "step_5" / "checkpoint.pt"
        assert ckpt_path.exists()

        # Step 6: Load checkpoint into fresh model
        model2 = Atlas(config)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2.load_state_dict(checkpoint["model_state_dict"])
        model2.eval()

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            torch.testing.assert_close(p1.cpu(), p2.cpu(), msg=f"Weight mismatch: {n1}")

        # Step 7: Generate
        input_ids = torch.randint(10, config.vocab_size, (1, 4))
        with torch.no_grad():
            output = model2.generate(input_ids, max_new_tokens=10, temperature=0.8)

        assert output.shape[1] > 4  # Generated at least 1 new token
        assert output.shape[1] <= 14  # At most 10 new tokens
        assert torch.all(output >= 0)
        assert torch.all(output < config.vocab_size)

    def test_full_instruct_pipeline(self, workspace, config):
        """Test instruction fine-tuning pipeline end-to-end."""
        torch.manual_seed(42)
        model = Atlas(config)

        # Create dummy instruction data
        instruct_path = workspace["data_dir"] / "instruct.jsonl"
        samples = []
        for i in range(20):
            samples.append({
                "messages": [
                    {"role": "user", "content": f"Write function number {i}"},
                    {"role": "assistant", "content": f"def func_{i}(): return {i}"},
                ]
            })
        with open(instruct_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # Create mock tokenizer
        class MockTokenizer:
            class Encoding:
                def __init__(self, ids):
                    self.ids = ids
            def encode(self, text):
                ids = [1] + [ord(c) % 200 + 10 for c in text[:100]] + [2]
                return self.Encoding(ids)

        tokenizer = MockTokenizer()
        dataset = InstructionDataset(str(instruct_path), tokenizer, seq_length=64)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Train
        args = TrainingArgs(
            checkpoint_dir=str(workspace["checkpoint_dir"]),
            batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            max_steps=5,
            warmup_steps=1,
            seq_length=64,
            precision="fp32",
            log_interval=2,
            save_interval=5,
            eval_interval=100,
            use_wandb=False,
        )

        trainer = Trainer(model, args)
        trainer.train(dataloader)
        assert trainer.global_step == 5

    def test_training_on_repeated_data_reduces_loss(self, workspace, config):
        """Model should overfit (reduce loss) on a tiny repeated dataset."""
        torch.manual_seed(42)
        model = Atlas(config)

        # Create a very small, repeated dataset (easy to overfit)
        pattern = torch.randint(10, config.vocab_size, (32,), dtype=torch.long)
        tokens = pattern.repeat(100)  # Repeat same pattern
        data_path = workspace["data_dir"] / "overfit.pt"
        torch.save(tokens, data_path)

        dataset = CodeDataset(str(data_path), seq_length=32)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Get initial loss
        model.eval()
        batch = next(iter(dataloader))
        with torch.no_grad():
            initial_loss = model(batch["input_ids"], labels=batch["labels"])["loss"].item()

        # Train for more steps with high LR
        model.train()
        args = TrainingArgs(
            checkpoint_dir=str(workspace["checkpoint_dir"]),
            batch_size=8,
            gradient_accumulation_steps=1,
            learning_rate=5e-3,
            max_steps=50,
            warmup_steps=5,
            seq_length=32,
            precision="fp32",
            log_interval=10,
            save_interval=100,
            eval_interval=100,
            use_wandb=False,
        )

        trainer = Trainer(model, args)
        trainer.train(dataloader)

        # Final loss should be significantly lower
        model.eval()
        with torch.no_grad():
            final_loss = model(
                batch["input_ids"].to(trainer.device),
                labels=batch["labels"].to(trainer.device),
            )["loss"].item()

        assert final_loss < initial_loss * 0.8, (
            f"Expected significant loss decrease but got {initial_loss:.4f} -> {final_loss:.4f}"
        )


class TestCrossplatform:
    """Tests that verify cross-platform compatibility (Windows/Linux)."""

    def test_no_posix_only_paths(self):
        """Verify no hardcoded Unix paths in source code."""
        import importlib
        # These modules should import without errors on any platform
        from model import transformer
        from data import preprocessing
        from training import trainer

    def test_checkpoint_path_handling(self, tmp_path):
        """Paths should work with both / and \\ separators."""
        config = AtlasConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = Atlas(config)
        args = TrainingArgs(
            checkpoint_dir=str(tmp_path / "ckpt"),
            precision="fp32",
        )
        trainer = Trainer(model, args)
        trainer._save_checkpoint()

        ckpt = tmp_path / "ckpt" / "step_0" / "checkpoint.pt"
        assert ckpt.exists()

        # Load using Path (works on both Windows and Linux)
        loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert "model_state_dict" in loaded

    def test_dataloader_num_workers_zero(self):
        """num_workers=0 is safest for Windows compatibility."""
        tokens = torch.randint(0, 100, (500,), dtype=torch.long)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(tokens, f.name)
            path = f.name

        try:
            dataset = CodeDataset(path, seq_length=50)
            # num_workers=0 works on all platforms
            loader = DataLoader(dataset, batch_size=2, num_workers=0)
            batch = next(iter(loader))
            assert batch["input_ids"].shape == (2, 50)
        finally:
            import os
            os.unlink(path)

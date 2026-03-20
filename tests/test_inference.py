"""
Tests for inference module.

Tests verify:
- Prompt formatting produces correct chat template
- AtlasInference loads model from checkpoint and generates text
- CLI and server entry points are importable
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from model.transformer import Atlas, AtlasConfig
from inference.server import AtlasInference


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Create a dummy checkpoint for testing."""
    config = AtlasConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Atlas(config)

    ckpt_dir = tmp_path / "test_checkpoint"
    ckpt_dir.mkdir()
    from dataclasses import asdict
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "args": {},
    }
    torch.save(checkpoint, ckpt_dir / "checkpoint.pt")
    return str(ckpt_dir)


@pytest.fixture
def dummy_tokenizer_file(tmp_path):
    """Create a minimal tokenizer for testing."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=256,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>",
                        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
                        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"],
    )
    # Train on some dummy text
    tokenizer.train_from_iterator(
        ["hello world", "fn main() { println!(\"hello\"); }",
         "package main\nimport \"fmt\"", "docker run -it ubuntu bash"],
        trainer=trainer,
    )

    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return str(path)


class TestPromptFormatting:
    def test_format_single_user_message(self, checkpoint_dir, dummy_tokenizer_file):
        engine = AtlasInference(checkpoint_dir, dummy_tokenizer_file, device="cpu")
        messages = [{"role": "user", "content": "Hello"}]
        prompt = engine.format_prompt(messages)
        assert "<|user|>" in prompt
        assert "Hello" in prompt
        assert prompt.endswith("<|assistant|>")

    def test_format_system_and_user(self, checkpoint_dir, dummy_tokenizer_file):
        engine = AtlasInference(checkpoint_dir, dummy_tokenizer_file, device="cpu")
        messages = [
            {"role": "system", "content": "You are a coder."},
            {"role": "user", "content": "Write Go code"},
        ]
        prompt = engine.format_prompt(messages)
        assert "<|system|>" in prompt
        assert "You are a coder." in prompt
        assert "<|user|>" in prompt
        assert "Write Go code" in prompt

    def test_format_multi_turn(self, checkpoint_dir, dummy_tokenizer_file):
        engine = AtlasInference(checkpoint_dir, dummy_tokenizer_file, device="cpu")
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Write code"},
        ]
        prompt = engine.format_prompt(messages)
        # Should have 2 user turns and 1 assistant turn, plus final assistant prefix
        assert prompt.count("<|user|>") == 2
        assert prompt.count("<|assistant|>") == 2  # 1 in history + 1 prefix for generation


class TestGeneration:
    def test_generate_returns_string(self, checkpoint_dir, dummy_tokenizer_file):
        engine = AtlasInference(checkpoint_dir, dummy_tokenizer_file, device="cpu")
        result = engine.generate("hello", max_new_tokens=5)
        assert isinstance(result, str)

    def test_chat_returns_string(self, checkpoint_dir, dummy_tokenizer_file):
        engine = AtlasInference(checkpoint_dir, dummy_tokenizer_file, device="cpu")
        messages = [{"role": "user", "content": "hello"}]
        result = engine.chat(messages, max_new_tokens=5)
        assert isinstance(result, str)

    def test_chat_strips_end_token(self, checkpoint_dir, dummy_tokenizer_file):
        engine = AtlasInference(checkpoint_dir, dummy_tokenizer_file, device="cpu")
        messages = [{"role": "user", "content": "hello"}]
        result = engine.chat(messages, max_new_tokens=10)
        assert "<|end|>" not in result

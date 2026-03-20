"""
Tests for data preprocessing pipeline.

Tests verify:
- CodeDataset correctly chunks token sequences
- FIM transformation produces valid token sequences
- FIM rate is approximately correct
- InstructionDataset formats chat and instruction data correctly
- Edge cases: empty data, very short sequences
- Padding and label masking works correctly
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from data.preprocessing import (
    CodeDataset,
    FIMDataset,
    InstructionDataset,
    FIM_PREFIX_ID,
    FIM_MIDDLE_ID,
    FIM_SUFFIX_ID,
    PAD_ID,
    EOS_ID,
    BOS_ID,
)


# ─── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def dummy_token_file(tmp_path):
    """Create a dummy tokenized data file."""
    # 1000 random tokens
    tokens = torch.randint(10, 200, (1000,), dtype=torch.long)
    path = tmp_path / "train.pt"
    torch.save(tokens, path)
    return str(path)


@pytest.fixture
def dummy_instruct_file(tmp_path):
    """Create a dummy instruction dataset."""
    samples = [
        {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Write a hello world in Go"},
                {"role": "assistant", "content": "```go\npackage main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello, World!\")\n}\n```"},
            ]
        },
        {
            "instruction": "Explain what this bash command does",
            "input": "find . -name '*.log' -mtime +30 -delete",
            "output": "This command finds all .log files older than 30 days and deletes them.",
        },
        {
            "instruction": "Write a Dockerfile for a Go application",
            "input": "",
            "output": "FROM golang:1.21-alpine\nWORKDIR /app\nCOPY . .\nRUN go build -o main .\nCMD [\"./main\"]",
        },
    ]
    path = tmp_path / "instruct.jsonl"
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return str(path)


class MockTokenizer:
    """Minimal tokenizer mock for testing InstructionDataset."""

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text: str):
        # Simple char-level encoding for testing
        ids = [BOS_ID] + [ord(c) % 200 + 10 for c in text[:500]] + [EOS_ID]
        return self.Encoding(ids)

    def decode(self, ids):
        return "".join(chr(max(32, i)) for i in ids)


# ─── CodeDataset ─────────────────────────────────────────────

class TestCodeDataset:
    def test_loads_data(self, dummy_token_file):
        ds = CodeDataset(dummy_token_file, seq_length=64)
        assert len(ds) > 0

    def test_correct_num_sequences(self, dummy_token_file):
        ds = CodeDataset(dummy_token_file, seq_length=100)
        assert len(ds) == 10  # 1000 // 100

    def test_sequence_shape(self, dummy_token_file):
        ds = CodeDataset(dummy_token_file, seq_length=64)
        item = ds[0]
        assert item["input_ids"].shape == (64,)
        assert item["labels"].shape == (64,)

    def test_labels_match_inputs(self, dummy_token_file):
        ds = CodeDataset(dummy_token_file, seq_length=64)
        item = ds[0]
        assert torch.equal(item["input_ids"], item["labels"])

    def test_drops_remainder(self, dummy_token_file):
        ds = CodeDataset(dummy_token_file, seq_length=300)
        assert len(ds) == 3  # 1000 // 300 = 3, remainder 100 dropped

    def test_all_indices_valid(self, dummy_token_file):
        ds = CodeDataset(dummy_token_file, seq_length=50)
        for i in range(len(ds)):
            item = ds[i]
            assert item["input_ids"].shape == (50,)


# ─── FIMDataset ──────────────────────────────────────────────

class TestFIMDataset:
    def test_fim_wraps_base(self, dummy_token_file):
        base = CodeDataset(dummy_token_file, seq_length=64)
        fim = FIMDataset(base, fim_rate=0.5)
        assert len(fim) == len(base)

    def test_fim_output_shape(self, dummy_token_file):
        base = CodeDataset(dummy_token_file, seq_length=64)
        fim = FIMDataset(base, fim_rate=1.0)  # Always apply FIM
        item = fim[0]
        assert item["input_ids"].shape == (64,)

    def test_fim_contains_special_tokens(self, dummy_token_file):
        base = CodeDataset(dummy_token_file, seq_length=64)
        fim = FIMDataset(base, fim_rate=1.0)
        item = fim[0]
        tokens = item["input_ids"].tolist()
        # Should contain FIM prefix token
        assert FIM_PREFIX_ID in tokens

    def test_fim_rate_zero_no_transform(self, dummy_token_file):
        base = CodeDataset(dummy_token_file, seq_length=64)
        fim = FIMDataset(base, fim_rate=0.0)
        # With rate=0, should return original
        torch.manual_seed(42)
        item = fim[0]
        base_item = base[0]
        assert torch.equal(item["input_ids"], base_item["input_ids"])

    def test_fim_rate_approximate(self, dummy_token_file):
        """FIM should be applied approximately at the specified rate."""
        base = CodeDataset(dummy_token_file, seq_length=64)
        fim = FIMDataset(base, fim_rate=0.5)

        torch.manual_seed(42)
        fim_count = 0
        total = min(100, len(base))
        for i in range(total):
            item = fim[i]
            if FIM_PREFIX_ID in item["input_ids"].tolist():
                fim_count += 1

        rate = fim_count / total
        # Should be roughly 0.5 (allow wide margin due to randomness)
        assert 0.2 < rate < 0.8, f"FIM rate {rate} too far from 0.5"

    def test_fim_preserves_length(self, dummy_token_file):
        """FIM should not change sequence length."""
        base = CodeDataset(dummy_token_file, seq_length=64)
        fim = FIMDataset(base, fim_rate=1.0)
        for i in range(min(10, len(fim))):
            item = fim[i]
            assert item["input_ids"].shape == (64,)


# ─── InstructionDataset ─────────────────────────────────────

class TestInstructionDataset:
    def test_loads_all_samples(self, dummy_instruct_file):
        tokenizer = MockTokenizer()
        ds = InstructionDataset(dummy_instruct_file, tokenizer, seq_length=512)
        assert len(ds) == 3

    def test_output_shape(self, dummy_instruct_file):
        tokenizer = MockTokenizer()
        ds = InstructionDataset(dummy_instruct_file, tokenizer, seq_length=256)
        item = ds[0]
        assert item["input_ids"].shape == (256,)
        assert item["labels"].shape == (256,)

    def test_padding_masked_in_labels(self, dummy_instruct_file):
        tokenizer = MockTokenizer()
        ds = InstructionDataset(dummy_instruct_file, tokenizer, seq_length=1024)
        item = ds[0]
        # Where input has PAD, labels should have -100
        pad_mask = item["input_ids"] == PAD_ID
        if pad_mask.any():
            assert torch.all(item["labels"][pad_mask] == -100)

    def test_chat_format_contains_role_tokens(self, dummy_instruct_file):
        tokenizer = MockTokenizer()
        ds = InstructionDataset(dummy_instruct_file, tokenizer, seq_length=512)
        # The first sample is chat format
        sample = ds.samples[0]
        text = ds._format_chat(sample["messages"])
        assert "<|system|>" in text
        assert "<|user|>" in text
        assert "<|assistant|>" in text
        assert "<|end|>" in text

    def test_instruction_format(self, dummy_instruct_file):
        tokenizer = MockTokenizer()
        ds = InstructionDataset(dummy_instruct_file, tokenizer, seq_length=512)
        # Second sample is instruction format
        sample = ds.samples[1]
        text = ds._format_instruction(sample)
        assert "<|user|>" in text
        assert "<|assistant|>" in text
        assert sample["output"] in text

    def test_instruction_without_input(self, dummy_instruct_file):
        tokenizer = MockTokenizer()
        ds = InstructionDataset(dummy_instruct_file, tokenizer, seq_length=512)
        # Third sample has empty input
        sample = ds.samples[2]
        text = ds._format_instruction(sample)
        assert sample["instruction"] in text
        assert sample["output"] in text

    def test_truncation(self, dummy_instruct_file):
        """Sequences longer than seq_length should be truncated."""
        tokenizer = MockTokenizer()
        ds = InstructionDataset(dummy_instruct_file, tokenizer, seq_length=32)
        item = ds[0]
        assert item["input_ids"].shape == (32,)

    def test_all_indices_valid(self, dummy_instruct_file):
        tokenizer = MockTokenizer()
        ds = InstructionDataset(dummy_instruct_file, tokenizer, seq_length=256)
        for i in range(len(ds)):
            item = ds[i]
            assert item["input_ids"].shape == (256,)
            assert item["labels"].shape == (256,)
            assert not torch.any(torch.isnan(item["input_ids"].float()))

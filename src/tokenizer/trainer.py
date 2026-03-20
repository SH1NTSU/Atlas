"""
Custom BPE tokenizer trainer for Atlas.

Trains a tokenizer optimized for: TypeScript/TSX, Go, Rust, Bash, DevOps configs.
Uses HuggingFace tokenizers library for fast BPE training.
"""

import json
from pathlib import Path
from typing import Optional

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders, processors


# Special tokens
SPECIAL_TOKENS = [
    "<pad>",      # 0
    "<s>",        # 1 — BOS
    "</s>",       # 2 — EOS
    "<unk>",      # 3
    "<|system|>", # 4
    "<|user|>",   # 5
    "<|assistant|>",  # 6
    "<|end|>",    # 7
    # FIM (Fill-in-the-Middle) tokens for code completion
    "<|fim_prefix|>",  # 8
    "<|fim_middle|>",  # 9
    "<|fim_suffix|>",  # 10
]


def create_tokenizer(vocab_size: int = 32000) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """Create an untrained BPE tokenizer with code-optimized settings."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Normalizer: NFC unicode normalization, no lowercasing
    tokenizer.normalizer = normalizers.NFC()

    # Pre-tokenizer: split on whitespace, digits, and punctuation
    # This is critical for code — we want meaningful token boundaries
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=pre_tokenizers.Regex(
                r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
            ),
            behavior="isolated",
            invert=False,
        ),
    ])

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    return tokenizer, trainer


def train_tokenizer(
    data_dir: str,
    output_path: str,
    vocab_size: int = 32000,
    max_files: Optional[int] = None,
) -> Tokenizer:
    """
    Train BPE tokenizer on code data.

    Args:
        data_dir: Directory containing .jsonl files from download_data.py
        output_path: Where to save the trained tokenizer
        vocab_size: Vocabulary size
        max_files: Max number of text samples to use (None = all)
    """
    data_path = Path(data_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Collect text samples from JSONL files
    def text_iterator():
        count = 0
        for jsonl_file in sorted(data_path.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    if max_files and count >= max_files:
                        return
                    record = json.loads(line)
                    yield record["text"]
                    count += 1

    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer, trainer = create_tokenizer(vocab_size)
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # Add post-processor for BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )

    # Enable padding
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")

    # Save
    tokenizer.save(str(output))
    print(f"Tokenizer saved to {output}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Load a trained tokenizer from file."""
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train BPE tokenizer for Atlas")
    parser.add_argument("--data-dir", default="data/raw/", help="Directory with .jsonl data files")
    parser.add_argument("--output", default="data/tokenizer.json", help="Output tokenizer path")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--max-files", type=int, default=None, help="Max samples to train on")
    args = parser.parse_args()

    train_tokenizer(args.data_dir, args.output, args.vocab_size, args.max_files)

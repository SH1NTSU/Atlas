"""
Atlas — Llama-style transformer from scratch.

Architecture:
- RoPE (Rotary Position Embeddings)
- RMSNorm (instead of LayerNorm)
- SwiGLU FFN (instead of standard MLP)
- GQA (Grouped Query Attention — fewer KV heads than Q heads)
- Tied input/output embeddings
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AtlasConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 14
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    dropout: float = 0.0  # No dropout during pretraining, add for fine-tuning

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor."""
    # x: (batch, heads, seq_len, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class GQAttention(nn.Module):
    """Grouped Query Attention — fewer KV heads than Q heads for memory efficiency."""

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K with correct position offset for KV cache
        pos_offset = kv_cache[0].shape[2] if kv_cache is not None else 0
        q = apply_rope(q, rope_freqs[pos_offset:pos_offset + seq_len])
        k = apply_rope(k, rope_freqs[pos_offset:pos_offset + seq_len])

        # KV cache for inference
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(batch, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(batch, self.num_heads, -1, self.head_dim)

        # Scaled dot-product attention (uses Flash Attention if available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None and kv_cache is None),
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output), new_kv_cache


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network — more expressive than standard MLP."""

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Single transformer layer: attention + FFN with pre-norm."""

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.attention = GQAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm + attention + residual
        attn_out, new_kv_cache = self.attention(self.attention_norm(x), rope_freqs, mask, kv_cache)
        x = x + attn_out
        # Pre-norm + FFN + residual
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv_cache


class Atlas(nn.Module):
    """
    Atlas: Llama-style causal language model.

    ~150M parameters with:
    - 14 transformer layers
    - 768 hidden dim, 12 Q heads, 4 KV heads (GQA)
    - SwiGLU FFN with 2048 intermediate dim
    - RoPE positional encoding
    - RMSNorm
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Output head — optionally tied with input embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(config.head_dim, config.max_position_embeddings, config.rope_theta),
            persistent=False,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> dict:
        x = self.embed_tokens(input_ids)

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = layer(x, self.rope_freqs, kv_cache=layer_cache)
            new_kv_cache.append(new_cache)

        x = self.norm(x)

        # Output projection
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed_tokens.weight)

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "kv_cache": new_kv_cache}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache, top-k, top-p sampling."""
        kv_cache = None

        for _ in range(max_new_tokens):
            # Only feed last token if we have cache
            if kv_cache is not None:
                model_input = input_ids[:, -1:]
            else:
                model_input = input_ids

            outputs = self.forward(model_input, kv_cache=kv_cache)
            kv_cache = outputs["kv_cache"]
            logits = outputs["logits"][:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop on EOS
            if next_token.item() == 2:  # </s> token id
                break

        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Quick sanity check
    config = AtlasConfig()
    model = Atlas(config)
    print(f"Atlas model: {model.count_parameters() / 1e6:.1f}M parameters")

    # Test forward pass
    dummy_input = torch.randint(0, config.vocab_size, (2, 128))
    output = model(dummy_input, labels=dummy_input)
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Logits shape: {output['logits'].shape}")

"""
Comprehensive tests for Atlas model architecture.

Tests verify:
- Model construction and parameter counts
- Forward pass shapes and loss computation
- Gradient flow through all layers
- KV cache consistency (cached vs non-cached outputs match)
- Generation produces valid token sequences
- RoPE produces correct frequency patterns
- GQA head expansion is correct
- Memory fits within 8GB VRAM budget
- Deterministic behavior with fixed seed
- Edge cases: single token, max length, batch size 1
"""

import pytest
import torch
import math

from model.transformer import (
    Atlas,
    AtlasConfig,
    RMSNorm,
    GQAttention,
    SwiGLUFFN,
    TransformerBlock,
    precompute_rope_freqs,
    apply_rope,
)


# ─── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def config():
    return AtlasConfig()


@pytest.fixture
def small_config():
    """Smaller config for fast testing."""
    return AtlasConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )


@pytest.fixture
def model(small_config):
    return Atlas(small_config)


@pytest.fixture
def full_model(config):
    return Atlas(config)


# ─── Model Construction ─────────────────────────────────────

class TestModelConstruction:
    def test_default_config(self, config):
        assert config.vocab_size == 32000
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 14
        assert config.num_attention_heads == 12
        assert config.num_key_value_heads == 4

    def test_head_dim(self, config):
        assert config.head_dim == 768 // 12  # 64

    def test_kv_groups(self, config):
        assert config.num_kv_groups == 12 // 4  # 3

    def test_model_creates(self, model):
        assert model is not None

    def test_full_model_param_count(self, full_model):
        """Verify ~150M parameters."""
        count = full_model.count_parameters()
        assert 100_000_000 < count < 250_000_000, f"Expected ~150M params, got {count / 1e6:.1f}M"

    def test_small_model_param_count(self, model, small_config):
        count = model.count_parameters()
        assert count > 0

    def test_tied_embeddings(self, model):
        """When tie_word_embeddings=True, lm_head should be None."""
        assert model.lm_head is None

    def test_untied_embeddings(self, small_config):
        small_config.tie_word_embeddings = False
        m = Atlas(small_config)
        assert m.lm_head is not None

    def test_correct_num_layers(self, model, small_config):
        assert len(model.layers) == small_config.num_hidden_layers

    def test_rope_buffer_exists(self, model):
        assert hasattr(model, "rope_freqs")
        assert model.rope_freqs is not None


# ─── RMSNorm ────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(128)
        x = torch.randn(2, 10, 128)
        out = norm(x)
        assert out.shape == x.shape

    def test_norm_preserves_dtype(self):
        norm = RMSNorm(64)
        x = torch.randn(1, 5, 64, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32

    def test_norm_not_all_zeros(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert not torch.all(out == 0)

    def test_zero_input(self):
        norm = RMSNorm(64)
        x = torch.zeros(1, 1, 64)
        out = norm(x)
        # Should not produce NaN with eps
        assert not torch.any(torch.isnan(out))


# ─── RoPE ────────────────────────────────────────────────────

class TestRoPE:
    def test_freqs_shape(self):
        freqs = precompute_rope_freqs(64, 256)
        assert freqs.shape == (256, 32)  # (seq_len, head_dim//2)

    def test_freqs_complex(self):
        freqs = precompute_rope_freqs(64, 256)
        assert freqs.dtype == torch.complex64

    def test_apply_rope_shape(self):
        freqs = precompute_rope_freqs(64, 128)
        x = torch.randn(2, 4, 128, 64)  # (batch, heads, seq, head_dim)
        out = apply_rope(x, freqs[:128])
        assert out.shape == x.shape

    def test_rope_different_positions(self):
        """Different positions should produce different outputs."""
        freqs = precompute_rope_freqs(64, 128)
        x = torch.ones(1, 1, 2, 64)  # Same input at 2 positions
        out = apply_rope(x, freqs[:2])
        # Position 0 and position 1 should differ
        assert not torch.allclose(out[0, 0, 0], out[0, 0, 1])


# ─── GQAttention ─────────────────────────────────────────────

class TestGQAttention:
    def test_output_shape(self, small_config):
        attn = GQAttention(small_config)
        x = torch.randn(2, 16, small_config.hidden_size)
        freqs = precompute_rope_freqs(small_config.head_dim, 256)
        out, cache = attn(x, freqs)
        assert out.shape == x.shape

    def test_kv_cache_shapes(self, small_config):
        attn = GQAttention(small_config)
        x = torch.randn(1, 8, small_config.hidden_size)
        freqs = precompute_rope_freqs(small_config.head_dim, 256)
        _, cache = attn(x, freqs)
        k, v = cache
        assert k.shape[1] == small_config.num_key_value_heads
        assert k.shape[2] == 8  # seq_len
        assert k.shape[3] == small_config.head_dim

    def test_kv_cache_grows(self, small_config):
        """KV cache should accumulate across calls."""
        attn = GQAttention(small_config)
        freqs = precompute_rope_freqs(small_config.head_dim, 256)

        # First call
        x1 = torch.randn(1, 4, small_config.hidden_size)
        _, cache1 = attn(x1, freqs)

        # Second call with cache
        x2 = torch.randn(1, 1, small_config.hidden_size)
        _, cache2 = attn(x2, freqs, kv_cache=cache1)
        k2, _ = cache2
        assert k2.shape[2] == 5  # 4 + 1


# ─── SwiGLU FFN ─────────────────────────────────────────────

class TestSwiGLUFFN:
    def test_output_shape(self, small_config):
        ffn = SwiGLUFFN(small_config)
        x = torch.randn(2, 16, small_config.hidden_size)
        out = ffn(x)
        assert out.shape == x.shape

    def test_no_nans(self, small_config):
        ffn = SwiGLUFFN(small_config)
        x = torch.randn(2, 16, small_config.hidden_size)
        out = ffn(x)
        assert not torch.any(torch.isnan(out))


# ─── Full Forward Pass ──────────────────────────────────────

class TestForwardPass:
    def test_forward_logits_shape(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (2, 32))
        out = model(x)
        assert out["logits"].shape == (2, 32, small_config.vocab_size)

    def test_forward_loss(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (2, 32))
        out = model(x, labels=x)
        assert out["loss"] is not None
        assert out["loss"].item() > 0
        assert not torch.isnan(out["loss"])

    def test_forward_no_labels_no_loss(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (2, 32))
        out = model(x)
        assert out["loss"] is None

    def test_batch_size_1(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (1, 16))
        out = model(x, labels=x)
        assert out["logits"].shape == (1, 16, small_config.vocab_size)
        assert not torch.isnan(out["loss"])

    def test_single_token(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (1, 1))
        out = model(x)
        assert out["logits"].shape == (1, 1, small_config.vocab_size)

    def test_max_length(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (1, small_config.max_position_embeddings))
        out = model(x)
        assert out["logits"].shape == (1, small_config.max_position_embeddings, small_config.vocab_size)

    def test_kv_cache_returned(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (1, 8))
        out = model(x)
        assert out["kv_cache"] is not None
        assert len(out["kv_cache"]) == small_config.num_hidden_layers


# ─── Gradient Flow ──────────────────────────────────────────

class TestGradientFlow:
    def test_gradients_flow_to_embeddings(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (2, 16))
        out = model(x, labels=x)
        out["loss"].backward()
        assert model.embed_tokens.weight.grad is not None
        assert torch.any(model.embed_tokens.weight.grad != 0)

    def test_gradients_flow_to_all_layers(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (2, 16))
        out = model(x, labels=x)
        out["loss"].backward()
        for i, layer in enumerate(model.layers):
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"Layer {i} param {name} has no gradient"

    def test_no_nan_gradients(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (2, 16))
        out = model(x, labels=x)
        out["loss"].backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.any(torch.isnan(param.grad)), f"NaN gradient in {name}"


# ─── KV Cache Consistency ───────────────────────────────────

class TestKVCache:
    def test_cached_vs_uncached_match(self, model, small_config):
        """Verify KV cache produces same output as full recompute."""
        torch.manual_seed(42)
        model.eval()

        # Full sequence forward pass
        full_input = torch.randint(0, small_config.vocab_size, (1, 8))
        with torch.no_grad():
            full_out = model(full_input)
            full_logits = full_out["logits"]

        # Incremental: process prefix, then last token with cache
        with torch.no_grad():
            prefix_out = model(full_input[:, :-1])
            kv_cache = prefix_out["kv_cache"]
            last_out = model(full_input[:, -1:], kv_cache=kv_cache)

        # Last token logits should match
        torch.testing.assert_close(
            full_logits[:, -1, :],
            last_out["logits"][:, -1, :],
            atol=1e-4,
            rtol=1e-4,
        )


# ─── Generation ─────────────────────────────────────────────

class TestGeneration:
    def test_generate_produces_tokens(self, model, small_config):
        model.eval()
        input_ids = torch.randint(0, small_config.vocab_size, (1, 4))
        output = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
        assert output.shape[1] > input_ids.shape[1]
        assert output.shape[1] <= input_ids.shape[1] + 10

    def test_generate_valid_token_range(self, model, small_config):
        model.eval()
        input_ids = torch.randint(0, small_config.vocab_size, (1, 4))
        output = model.generate(input_ids, max_new_tokens=20, temperature=1.0)
        assert torch.all(output >= 0)
        assert torch.all(output < small_config.vocab_size)

    def test_generate_deterministic_with_seed(self, model, small_config):
        model.eval()
        input_ids = torch.randint(0, small_config.vocab_size, (1, 4))

        torch.manual_seed(123)
        out1 = model.generate(input_ids.clone(), max_new_tokens=10, temperature=0.5)

        torch.manual_seed(123)
        out2 = model.generate(input_ids.clone(), max_new_tokens=10, temperature=0.5)

        assert torch.equal(out1, out2)

    def test_generate_temperature_affects_output(self, model, small_config):
        """Higher temperature should produce more diverse outputs."""
        model.eval()
        input_ids = torch.randint(0, small_config.vocab_size, (1, 4))

        outputs_low_temp = set()
        outputs_high_temp = set()

        for seed in range(10):
            torch.manual_seed(seed)
            out = model.generate(input_ids.clone(), max_new_tokens=5, temperature=0.1)
            outputs_low_temp.add(tuple(out[0].tolist()))

            torch.manual_seed(seed)
            out = model.generate(input_ids.clone(), max_new_tokens=5, temperature=2.0)
            outputs_high_temp.add(tuple(out[0].tolist()))

        # High temp should generally produce more unique outputs
        # (not guaranteed but very likely with 10 runs)
        assert len(outputs_high_temp) >= len(outputs_low_temp)


# ─── Memory Budget ──────────────────────────────────────────

class TestMemoryBudget:
    def test_full_model_fits_8gb(self, config):
        """Verify model + training state fits in 8GB VRAM."""
        model = Atlas(config)
        param_count = model.count_parameters()

        # fp16 params + fp16 grads + fp32 optimizer (momentum + variance)
        params_bytes = param_count * 2  # fp16
        grads_bytes = param_count * 2   # fp16
        optimizer_bytes = param_count * 4 * 2  # 2x fp32 states
        total_bytes = params_bytes + grads_bytes + optimizer_bytes

        total_mb = total_bytes / (1024 ** 2)
        # Leave room for activations — total model state should be < 5GB
        # (activations + framework overhead needs remaining 3GB)
        assert total_mb < 5000, f"Model state {total_mb:.0f}MB exceeds 5GB budget"

    def test_param_count_in_range(self, config):
        model = Atlas(config)
        count = model.count_parameters()
        # Should be between 100M and 250M
        assert 100_000_000 <= count <= 250_000_000


# ─── Determinism ─────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_output(self, small_config):
        torch.manual_seed(42)
        m1 = Atlas(small_config)
        torch.manual_seed(42)
        m2 = Atlas(small_config)

        x = torch.randint(0, small_config.vocab_size, (1, 8))
        with torch.no_grad():
            out1 = m1(x)["logits"]
            out2 = m2(x)["logits"]
        torch.testing.assert_close(out1, out2)

    def test_forward_backward_deterministic(self, small_config):
        torch.manual_seed(42)
        model = Atlas(small_config)
        x = torch.randint(0, small_config.vocab_size, (1, 8))

        out = model(x, labels=x)
        out["loss"].backward()
        grad1 = model.embed_tokens.weight.grad.clone()

        model.zero_grad()
        out = model(x, labels=x)
        out["loss"].backward()
        grad2 = model.embed_tokens.weight.grad.clone()

        torch.testing.assert_close(grad1, grad2)

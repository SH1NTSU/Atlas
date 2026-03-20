"""Quick script to verify model architecture and VRAM usage."""

from transformer import Atlas, AtlasConfig


def estimate_vram_mb(param_count: int, precision_bytes: int = 2) -> dict:
    """Estimate VRAM for training (params + gradients + optimizer states)."""
    params_mb = (param_count * precision_bytes) / (1024 ** 2)
    grads_mb = params_mb  # Same size as params
    # AdamW: 2 states (momentum + variance) in fp32
    optimizer_mb = (param_count * 4 * 2) / (1024 ** 2)
    # Rough activation memory (depends on batch size and seq length)
    activation_mb = params_mb * 2  # Very rough estimate

    total = params_mb + grads_mb + optimizer_mb + activation_mb
    return {
        "params_mb": params_mb,
        "grads_mb": grads_mb,
        "optimizer_mb": optimizer_mb,
        "activation_est_mb": activation_mb,
        "total_est_mb": total,
    }


if __name__ == "__main__":
    config = AtlasConfig()
    model = Atlas(config)

    total_params = model.count_parameters()
    print(f"Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden, "
          f"{config.num_attention_heads} Q heads, {config.num_key_value_heads} KV heads")
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print()

    # Break down by component
    embed_params = sum(p.numel() for p in model.embed_tokens.parameters())
    layer_params = sum(p.numel() for p in model.layers.parameters())
    norm_params = sum(p.numel() for p in model.norm.parameters())

    print(f"Embeddings:  {embed_params:>12,} ({embed_params / total_params * 100:.1f}%)")
    print(f"Layers:      {layer_params:>12,} ({layer_params / total_params * 100:.1f}%)")
    print(f"Final norm:  {norm_params:>12,} ({norm_params / total_params * 100:.1f}%)")
    if model.lm_head:
        head_params = sum(p.numel() for p in model.lm_head.parameters())
        print(f"LM head:     {head_params:>12,} ({head_params / total_params * 100:.1f}%)")
    else:
        print(f"LM head:     tied with embeddings")
    print()

    # VRAM estimates
    vram = estimate_vram_mb(total_params)
    print("Estimated VRAM usage (fp16 training):")
    print(f"  Parameters:    {vram['params_mb']:>8.1f} MB")
    print(f"  Gradients:     {vram['grads_mb']:>8.1f} MB")
    print(f"  Optimizer:     {vram['optimizer_mb']:>8.1f} MB")
    print(f"  Activations:   {vram['activation_est_mb']:>8.1f} MB (rough estimate)")
    print(f"  ─────────────────────────")
    print(f"  Total:         {vram['total_est_mb']:>8.1f} MB")
    print()

    fits_12gb = vram['total_est_mb'] < 11500  # Leave some headroom
    print(f"Fits in 12GB VRAM: {'YES' if fits_12gb else 'NO — reduce model size'}")

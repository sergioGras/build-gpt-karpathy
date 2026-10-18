"""Shape and parameter arithmetic for the GPT-2 architecture."""

from .config import ModelConfig


def gpt2_parameter_breakdown(config: ModelConfig) -> dict[str, int]:
    """Count GPT-2 parameters without constructing tensors or allocating a model.

    Assumptions follow build-nanogpt: fused QKV and linear layers have biases,
    LayerNorms have scale and shift, and the output head shares token embeddings.
    """
    vocab = config.vocab_size
    context = config.context_length
    width = config.n_embd
    layers = config.n_layer
    return {
        "token_embeddings": vocab * width,
        "position_embeddings": context * width,
        "qkv_all_blocks": layers * (3 * width * width + 3 * width),
        "attention_projection_all_blocks": layers * (width * width + width),
        "mlp_expand_all_blocks": layers * (4 * width * width + 4 * width),
        "mlp_project_all_blocks": layers * (4 * width * width + width),
        "layer_norms_all_blocks": layers * 4 * width,
        "final_layer_norm": 2 * width,
        # lm_head.weight is the same parameter as the token embedding table.
        "additional_tied_output_head": 0,
    }

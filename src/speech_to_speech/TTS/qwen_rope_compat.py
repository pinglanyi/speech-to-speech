"""Bridge qwen-tts-hf's legacy default RoPE initializer to modern configs."""
from functools import wraps
from types import SimpleNamespace


def install_qwen_rope_compat(registry=None) -> bool:
    if registry is None:
        # faster-qwen3-tts imports qwen_tts lazily inside from_pretrained.
        # Register its shim now so adaptation happens before model construction.
        from qwen_tts._transformers_compat import patch_transformers_rope_registry
        patch_transformers_rope_registry()
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        registry = ROPE_INIT_FUNCTIONS
    original = registry.get("default")
    # Only adapt the known Qwen shim. Preserve upstream/native initializers.
    if (getattr(original, "__module__", "") != "qwen_tts._transformers_compat"
            or getattr(original, "_s2s_rope_compat", False)):
        return False

    @wraps(original)
    def compatible(config, device=None, seq_len=None, layer_type=None):
        if not hasattr(config, "rope_theta"):
            parameters = getattr(config, "rope_parameters", None) or {}
            if layer_type is not None and layer_type in parameters:
                parameters = parameters[layer_type]
            if "rope_theta" not in parameters:
                raise ValueError("Model config has no rope_theta in rope_parameters")
            config = SimpleNamespace(
                rope_theta=parameters["rope_theta"],
                head_dim=getattr(config, "head_dim", None),
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                partial_rotary_factor=parameters.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)),
            )
        return original(config, device=device, seq_len=seq_len, layer_type=layer_type)

    compatible._s2s_rope_compat = True
    registry["default"] = compatible
    return True

"""Map a checkpoint's config.model_type to the model class and pipeline settings.

Kept free of transformers imports at module level so the mapping is testable
without the training environment; the class itself is resolved lazily.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    class_name: str      # transformers class implementing the checkpoint
    rope_type: str       # which get_rope_index variant the dataset uses
    packing_mode: str    # how packed micro-batches reach the model (see data_processor)
    # Whether the chat template accepts enable_thinking (Qwen3.5/3.8 inject a
    # reasoning instruction into every sample unless it is disabled)
    thinking_template: bool = False


# packing_mode:
#   "attention_mask": legacy path — the collator puts cu_seqlens in attention_mask
#       and trainer.replace_qwen2_vl_attention_class() patches the attention to use it.
#   "fa_kwargs": native path — the collator emits HF FlashAttentionKwargs
#       (cu_seq_lens_q/k, max_length_q/k); the model's attention and Gated DeltaNet
#       layers consume them directly (requires attn_implementation=flash_attention_2).
MODEL_REGISTRY = {
    "qwen2_vl": ModelSpec("Qwen2VLForConditionalGeneration", "qwen2vl", "attention_mask"),
    "qwen2_5_vl": ModelSpec("Qwen2_5_VLForConditionalGeneration", "qwen2.5vl", "attention_mask"),
    "qwen3_vl": ModelSpec("Qwen3VLForConditionalGeneration", "qwen3vl", "attention_mask"),
    "qwen3_vl_moe": ModelSpec("Qwen3VLMoeForConditionalGeneration", "qwen3vl", "attention_mask"),
    # Qwen3.5 / Qwen3.8 dense VLMs: hybrid Gated-DeltaNet + full-attention text
    # stack with a Qwen3-VL-style vision tower; M-RoPE indexing is identical to
    # Qwen3-VL's, so the dataset reuses that rope variant.
    "qwen3_5": ModelSpec("Qwen3_5ForConditionalGeneration", "qwen3vl", "fa_kwargs", thinking_template=True),
}


def resolve_model_spec(model_type: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_type]
    except KeyError:
        raise ValueError(
            f"unsupported model_type {model_type!r}; known: {sorted(MODEL_REGISTRY)}"
        ) from None


def resolve_model_class(spec: ModelSpec):
    import transformers

    cls = getattr(transformers, spec.class_name, None)
    if cls is None:
        raise ImportError(
            f"transformers {transformers.__version__} has no {spec.class_name}; "
            "this checkpoint needs a newer transformers (Qwen3.5/3.8 require >= 5.8)."
        )
    return cls

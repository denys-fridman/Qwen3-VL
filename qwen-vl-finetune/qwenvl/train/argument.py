import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_llm_last_n_layers: int = field(
        default=-1,
        metadata={
            "help": "With tune_mm_llm, train only the last N decoder layers (plus "
            "final norm and lm_head) and freeze the rest, cutting optimizer-state "
            "memory for development runs. -1 trains the full LLM."
        },
    )

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    train_on_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Compute next-token loss on the full sequence (continued pretraining) "
            "instead of assistant responses only (SFT). Vision placeholder tokens are always excluded."
        },
    )
    allow_text_only: bool = field(
        default=False,
        metadata={
            "help": "Allow samples without images: run a zero-weighted dummy vision "
            "forward on text-only batches so DeepSpeed ZeRO-3 collectives stay "
            "aligned across ranks."
        },
    )
    require_image_per_batch: bool = field(
        default=False,
        metadata={
            "help": "Build train batches with >=1 image sample each (and keep the "
            "eval split image-only), so text-only data can be mixed in safely even "
            "with a trainable vision tower under ZeRO (S1-style training)."
        },
    )
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    eval_samples: int = field(
        default=0,
        metadata={
            "help": "Hold out this many samples (fixed shuffle) as an eval set; "
            "0 disables evaluation."
        },
    )
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)

# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import (
    replace_qwen2_vl_attention_class,
    enable_dummy_vision_forward,
    enable_fa_kwargs_packing,
    enable_image_guaranteed_batches,
)

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.model_registry import resolve_model_class, resolve_model_spec
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# transformers v5 removed the deprecated .visual/.language_model proxies from
# the ForConditionalGeneration wrapper; they only exist on the inner .model.
def get_visual(model):
    return model.visual if hasattr(model, "visual") else model.model.visual


def get_language_model(model):
    if hasattr(model, "language_model"):
        return model.language_model
    return model.model.language_model


def set_model(model_args, model):
    visual = get_visual(model)
    language_model = get_language_model(model)

    if model_args.tune_mm_vision:
        for n, p in visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in language_model.named_parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
        if model_args.tune_llm_last_n_layers > 0:
            for n, p in language_model.named_parameters():
                p.requires_grad = False
            for layer in language_model.layers[-model_args.tune_llm_last_n_layers :]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in language_model.norm.parameters():
                p.requires_grad = True
    else:
        for n, p in language_model.named_parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if (
        data_args.allow_text_only
        and model_args.tune_mm_vision
        and not data_args.require_image_per_batch
    ):
        raise ValueError(
            "allow_text_only is incompatible with tune_mm_vision: the dummy vision "
            "forward keeps ZeRO-3 collectives aligned only while the vision tower is "
            "frozen. With a trainable tower, ranks with real images produce deepstack-"
            "merger gradients interleaved with LLM-layer gradients, while text-only "
            "ranks produce all vision gradients after the LLM backward, so gradient "
            "reduce-scatter order diverges across ranks and NCCL hangs. To train the "
            "vision tower on data with text-only samples, set require_image_per_batch "
            "True; otherwise use an image-only dataset and set allow_text_only False."
        )

    # Pick the model class from the checkpoint's config.model_type (not from the
    # path name); the spec also fixes the rope variant and packing path.
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir
    )
    spec = resolve_model_spec(config.model_type)
    model = resolve_model_class(spec).from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    data_args.model_type = spec.rope_type
    data_args.packing_mode = spec.packing_mode
    # vision placeholder ids differ between vocabularies (Qwen3-VL vs Qwen3.5/3.8)
    data_args.image_token_id = getattr(config, "image_token_id", 151655)
    data_args.video_token_id = getattr(config, "video_token_id", 151656)
    data_args.vision_start_token_id = getattr(config, "vision_start_token_id", 151652)
    # Qwen3.5/3.8 chat templates inject a reasoning instruction unless thinking is off
    data_args.chat_template_kwargs = {"enable_thinking": False} if spec.thinking_template else {}

    print(
        f"the initialized model is {model_args.model_name_or_path}: class "
        f"{model.__class__.__name__}, model_type {config.model_type}, "
        f"packing {spec.packing_mode}"
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        if spec.packing_mode == "attention_mask":
            replace_qwen2_vl_attention_class()
        else:
            # hybrid stacks (Qwen3.5/3.8) need the varlen Gated DeltaNet kernel
            # for packing to respect document boundaries
            text_config = getattr(config, "text_config", config)
            has_linear_attention = "linear_attention" in (getattr(text_config, "layer_types", None) or [])
            enable_fa_kwargs_packing(check_gdn_kernel=has_linear_attention)
    if data_args.allow_text_only:
        enable_dummy_vision_forward()
    if data_args.require_image_per_batch:
        enable_image_guaranteed_batches()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            get_visual(model).print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    # let the dataset reject samples that tokenize beyond the context length
    # (truncating them would break image/pixel alignment)
    data_args.model_max_length = training_args.model_max_length

    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

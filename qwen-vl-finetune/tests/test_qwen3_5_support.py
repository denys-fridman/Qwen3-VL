"""Unit tests for the Qwen3.5/3.8 (qwen3_5) support path.

Runs anywhere with torch: transformers and flash_attn are replaced by stub
modules, so only our own data/packing/dispatch logic is exercised.

    cd qwen-vl-finetune && python -m unittest tests.test_qwen3_5_support -v
"""

import importlib.abc
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "qwenvl" / "train"))  # train_qwen.py does `from trainer import ...`


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve MagicMock modules for third-party packages absent from the test box."""

    PREFIXES = ("transformers", "flash_attn")

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.PREFIXES:
            return importlib.util.spec_from_loader(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = mock.MagicMock(name=spec.name)
        module.__path__ = []
        module.__spec__ = spec
        return module

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubFinder())
for _name in [m for m in sys.modules if m.split(".")[0] in _StubFinder.PREFIXES]:
    del sys.modules[_name]

from qwenvl.data import rope2d  # noqa: E402
from qwenvl.data.data_processor import (  # noqa: E402
    IGNORE_INDEX,
    FlattenedDataCollatorForSupervisedDataset,
    preprocess_qwen_visual,
)
from qwenvl.train import trainer  # noqa: E402
from qwenvl.train.model_registry import MODEL_REGISTRY, resolve_model_spec  # noqa: E402

# Qwen3-VL ids (rope2d defaults) and Qwen3.8-27B ids (from its config.json)
QWEN3VL = dict(image=151655, video=151656, vision_start=151652, vision_end=151653)
QWEN38 = dict(image=248056, video=248057, vision_start=248053, vision_end=248054)


class RegistryTest(unittest.TestCase):
    def test_qwen3_5_spec(self):
        spec = resolve_model_spec("qwen3_5")
        self.assertEqual(spec.class_name, "Qwen3_5ForConditionalGeneration")
        self.assertEqual(spec.rope_type, "qwen3vl")  # identical get_rope_index
        self.assertEqual(spec.packing_mode, "fa_kwargs")
        self.assertTrue(spec.thinking_template)

    def test_qwen_vl_specs_keep_legacy_packing(self):
        for model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"):
            spec = resolve_model_spec(model_type)
            self.assertEqual(spec.packing_mode, "attention_mask", model_type)
            self.assertFalse(spec.thinking_template, model_type)
        self.assertEqual(resolve_model_spec("qwen3_vl").class_name, "Qwen3VLForConditionalGeneration")
        self.assertEqual(resolve_model_spec("qwen3_vl_moe").class_name, "Qwen3VLMoeForConditionalGeneration")

    def test_unknown_model_type(self):
        with self.assertRaises(ValueError):
            resolve_model_spec("llama")
        self.assertEqual(len(MODEL_REGISTRY), 5)


def _sequence(ids, n_text_before=3, n_text_after=4, image_tokens=4):
    """<text> <vision_start> <image_pad>*k <vision_end> <text>"""
    text = list(range(1000, 1000 + n_text_before))
    tail = list(range(2000, 2000 + n_text_after))
    return torch.tensor([text + [ids["vision_start"]] + [ids["image"]] * image_tokens + [ids["vision_end"]] + tail])


class RopeTokenIdTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(rope2d.set_token_ids, QWEN3VL["image"], QWEN3VL["video"], QWEN3VL["vision_start"])

    def test_positions_identical_across_vocabularies(self):
        grid = torch.tensor([[1, 4, 4]])  # 16 patches -> 4 tokens after 2x2 merge
        ref, _ = rope2d.get_rope_index_3(2, _sequence(QWEN3VL), image_grid_thw=grid)

        rope2d.set_token_ids(QWEN38["image"], QWEN38["video"], QWEN38["vision_start"])
        got, _ = rope2d.get_rope_index_3(2, _sequence(QWEN38), image_grid_thw=grid)

        self.assertEqual(ref.shape, (3, 1, 13))
        self.assertTrue(torch.equal(ref, got))
        # image tokens occupy positions 4..5 on h/w (2x2 grid), text resumes after
        t, h, w = got[:, 0, :]
        self.assertEqual(h[4:8].tolist(), [4, 4, 5, 5])
        self.assertEqual(w[4:8].tolist(), [4, 5, 4, 5])
        self.assertEqual(t[8].item(), 6)  # <vision_end> follows max(image position)+1
        self.assertEqual(t[-1].item(), 10)

    def test_qwen3vl_ids_not_recognised_after_switch(self):
        grid = torch.tensor([[1, 4, 4]])
        rope2d.set_token_ids(QWEN38["image"], QWEN38["video"], QWEN38["vision_start"])
        # Qwen3-VL placeholders are now plain text: positions are 0..12 on all axes
        pos, _ = rope2d.get_rope_index_3(2, _sequence(QWEN3VL), image_grid_thw=grid)
        self.assertEqual(pos[0, 0].tolist(), list(range(13)))
        self.assertTrue(torch.equal(pos[0], pos[1]))


class CollatorTest(unittest.TestCase):
    def _instances(self):
        out = []
        for length, start in ((5, 0), (3, 100)):
            # same shapes as LazySupervisedDataset._get_item: (1, L) ids/labels,
            # (3, 1, L) M-RoPE positions, attention_mask = [L]
            out.append(
                dict(
                    input_ids=torch.arange(start, start + length).unsqueeze(0),
                    labels=torch.arange(start, start + length).unsqueeze(0),
                    attention_mask=[length],
                    position_ids=torch.arange(length).expand(3, 1, length),
                )
            )
        out[0]["pixel_values"] = torch.zeros(4, 8)
        out[0]["image_grid_thw"] = torch.tensor([[1, 2, 2]])
        return out

    def test_fa_kwargs_mode(self):
        collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=None, packing_mode="fa_kwargs")
        batch = collator(self._instances())
        self.assertIsNone(batch["attention_mask"])
        self.assertEqual(batch["cu_seq_lens_q"].tolist(), [0, 5, 8])
        self.assertEqual(batch["cu_seq_lens_q"].dtype, torch.int32)
        self.assertTrue(torch.equal(batch["cu_seq_lens_q"], batch["cu_seq_lens_k"]))
        self.assertEqual(batch["max_length_q"], 5)
        self.assertEqual(batch["max_length_k"], 5)
        self.assertIsInstance(batch["max_length_q"], int)
        self.assertEqual(batch["input_ids"].shape, (1, 8))
        self.assertEqual(batch["position_ids"].shape, (3, 1, 8))
        self.assertEqual(batch["pixel_values"].shape, (4, 8))
        self.assertEqual(batch["image_grid_thw"].tolist(), [[1, 2, 2]])
        self.assertIsNone(batch["pixel_values_videos"])

    def test_legacy_mode_unchanged(self):
        collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=None)
        batch = collator(self._instances())
        self.assertEqual(batch["attention_mask"].tolist(), [0, 5, 8])
        for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
            self.assertNotIn(key, batch)


class PreprocessTest(unittest.TestCase):
    def _processor(self, ids):
        processor = mock.MagicMock()
        seq = _sequence(ids, n_text_before=2, n_text_after=2, image_tokens=2)
        processor.apply_chat_template.return_value = {
            "input_ids": seq,
            "pixel_values": torch.zeros(8, 8),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
        }
        return processor, seq

    def test_masks_config_token_ids_and_forwards_template_kwargs(self):
        processor, seq = self._processor(QWEN38)
        out = preprocess_qwen_visual(
            [{"conversations": [{"from": "human", "value": "<image>\nhello"}], "image": "x.jpg"}],
            processor,
            train_on_all_tokens=True,
            image_token_id=QWEN38["image"],
            video_token_id=QWEN38["video"],
            chat_template_kwargs={"enable_thinking": False},
        )
        labels = out["labels"]
        masked = labels == IGNORE_INDEX
        self.assertTrue(torch.equal(masked, seq == QWEN38["image"]))
        # <vision_start>/<vision_end> and text remain valid targets
        self.assertEqual(int(masked.sum()), 2)
        kwargs = processor.apply_chat_template.call_args.kwargs
        self.assertIs(kwargs["enable_thinking"], False)
        self.assertTrue(kwargs["tokenize"])

    def test_default_ids_still_mask_qwen3vl_placeholders(self):
        processor, seq = self._processor(QWEN3VL)
        out = preprocess_qwen_visual(
            [{"conversations": [{"from": "human", "value": "<image>\nhello"}], "image": "x.jpg"}],
            processor,
            train_on_all_tokens=True,
        )
        self.assertTrue(torch.equal(out["labels"] == IGNORE_INDEX, seq == QWEN3VL["image"]))
        self.assertNotIn("enable_thinking", processor.apply_chat_template.call_args.kwargs)


class PackingKwargsTest(unittest.TestCase):
    def test_strip_packing_kwargs(self):
        seen = {}

        class Vision:
            def forward(self, hidden_states, grid_thw, **kwargs):
                seen.update(kwargs)
                return hidden_states

        Vision.forward = trainer._strip_packing_kwargs(Vision.forward)
        Vision().forward(
            torch.zeros(1), torch.zeros(1), cu_seq_lens_q=1, cu_seq_lens_k=1, max_length_q=1, max_length_k=1, return_dict=True
        )
        self.assertEqual(seen, {"return_dict": True})

    def test_kernel_guard(self):
        with mock.patch.dict(sys.modules, {"fla": None}):  # import fails
            self.assertFalse(trainer.varlen_gdn_kernel_available())
            with mock.patch.dict(os.environ, {}, clear=True):
                os.environ.pop("ALLOW_GDN_TORCH_FALLBACK", None)
                with self.assertRaises(RuntimeError):
                    trainer.enable_fa_kwargs_packing(check_gdn_kernel=True)
            with mock.patch.dict(os.environ, {"ALLOW_GDN_TORCH_FALLBACK": "1"}):
                trainer.enable_fa_kwargs_packing(check_gdn_kernel=True)  # warns only
        fla = types.ModuleType("fla")
        ops = types.ModuleType("fla.ops")
        gdr = types.ModuleType("fla.ops.gated_delta_rule")
        gdr.chunk_gated_delta_rule = lambda *a, **k: None
        with mock.patch.dict(sys.modules, {"fla": fla, "fla.ops": ops, "fla.ops.gated_delta_rule": gdr}):
            self.assertTrue(trainer.varlen_gdn_kernel_available())
            trainer.enable_fa_kwargs_packing(check_gdn_kernel=True)


if __name__ == "__main__":
    unittest.main()

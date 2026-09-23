#!/usr/bin/env python3
"""Report whether the training environment has what a checkpoint needs.

Prints package versions and whether the optional kernels / model classes are
importable. With --require, exits non-zero if any named check is missing.

    python tools/check_env.py
    python tools/check_env.py --require qwen3_5 fla causal_conv1d
"""

import argparse
import importlib
import sys


def probe_version(module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as error:  # noqa: BLE001 - report any import failure
        return None, f"{type(error).__name__}: {error}"[:120]
    return getattr(module, "__version__", "present"), None


def probe_attr(module_name, attr):
    try:
        return hasattr(importlib.import_module(module_name), attr), None
    except Exception as error:  # noqa: BLE001
        return False, f"{type(error).__name__}: {error}"[:120]


CHECKS = {
    # name: (description, probe)
    "torch": ("PyTorch", lambda: probe_version("torch")),
    "transformers": ("transformers", lambda: probe_version("transformers")),
    "deepspeed": ("DeepSpeed", lambda: probe_version("deepspeed")),
    "flash_attn": ("flash-attn (attention kernels)", lambda: probe_version("flash_attn")),
    "fla": ("flash-linear-attention (Gated DeltaNet kernels; Qwen3.5/3.8)", lambda: probe_version("fla")),
    "causal_conv1d": ("causal-conv1d (Gated DeltaNet conv kernels; Qwen3.5/3.8)", lambda: probe_version("causal_conv1d")),
    "kernels": ("HF kernels hub client (alternative kernel source)", lambda: probe_version("kernels")),
    "qwen3_vl": ("Qwen3VLForConditionalGeneration", lambda: probe_attr("transformers", "Qwen3VLForConditionalGeneration")),
    "qwen3_5": ("Qwen3_5ForConditionalGeneration (Qwen3.5/3.8; transformers >= 5.8)", lambda: probe_attr("transformers", "Qwen3_5ForConditionalGeneration")),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", nargs="*", default=[], choices=sorted(CHECKS), help="checks that must pass")
    args = parser.parse_args()

    missing = []
    print("=== environment check ===")
    for name, (description, probe) in CHECKS.items():
        result, error = probe()
        ok = bool(result)
        status = "OK     " if ok else "MISSING"
        detail = result if isinstance(result, str) else ("" if ok else (error or ""))
        print(f"{status} {name:14s} {description}{'  [' + detail + ']' if detail else ''}")
        if not ok and name in args.require:
            missing.append(name)
    print("=========================")
    if missing:
        print(f"required checks missing: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

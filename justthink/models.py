"""Model registry. Layers come from the Pain Axis release: the steering layer is the one
their S2 ladder used, and the monitor layer is the layer the vector was extracted at."""
import os

import numpy as np
import torch

from .paths import MODELS, pain_vector_file

REGISTRY = {
    # Main study, on Kaggle (2x T4). T4 GPUs have no bfloat16, so the weights run in float16.
    "qwen7b-fp16": dict(backend="torch", hf="Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28",
                        pain_axis="Qwen_2.5_7B_instruct", steer_layer=16, monitor_layer=24),
    # Second-family replication (PREREGISTRATION_LLAMA.md). The weights are Meta's Llama 3.1 8B
    # Instruct from the ungated unsloth mirror, so no Hugging Face licence token is needed on
    # Kaggle. Layers follow Pain Axis: S2 steering at block 16, extraction at block 28.
    "llama8b-fp16": dict(backend="torch", hf="unsloth/Llama-3.1-8B-Instruct", revision="4699cc75b550f9c6f3173fb80f4703b62d946aa5",
                         pain_axis="Llama_3.1_8B_instruct", steer_layer=16, monitor_layer=28,
                         fmt="llama31", eot="<|eot_id|>", also_end=("<|eom_id|>",), hidden_bos=True),
    # The local MLX setup the main study was first started on (see PREREGISTRATION.md, amendment 1).
    "qwen7b-mlx4": dict(backend="mlx", path=MODELS / "Qwen2.5-7B-Instruct-4bit",
                        hf="mlx-community/Qwen2.5-7B-Instruct-4bit", revision="c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
                        pain_axis="Qwen_2.5_7B_instruct", steer_layer=16, monitor_layer=24),
    # Tiny models for smoke tests only.
    "llama1b-test": dict(backend="torch", hf="unsloth/Llama-3.2-1B-Instruct", pain_axis=None, d=2048,
                         steer_layer=8, monitor_layer=12, fmt="llama31", eot="<|eot_id|>",
                         also_end=("<|eom_id|>",), hidden_bos=True),
    "qwen05b-cuda": dict(backend="torch", hf="Qwen/Qwen2.5-0.5B-Instruct", pain_axis=None,
                         steer_layer=8, monitor_layer=12),
    "qwen05b-test": dict(backend="mlx", path="Qwen/Qwen2.5-0.5B-Instruct", pain_axis=None,
                         steer_layer=8, monitor_layer=12),
}


def released_pain_vector(name):
    cfg = REGISTRY[name]
    if cfg["pain_axis"] is None:
        return np.random.default_rng(0).normal(size=cfg.get("d", 896)).astype(np.float32)
    d = torch.load(pain_vector_file(cfg["pain_axis"]), map_location="cpu", weights_only=False)
    assert int(d["layer"]) == cfg["monitor_layer"], "monitor layer should be the extraction layer"
    return d["s2_pain_vector"].float().numpy()


def load(name):
    cfg = REGISTRY[name]
    v = released_pain_vector(name)
    if cfg["backend"] == "mlx":
        from .backend_mlx import Steerer
        return Steerer(cfg["path"], cfg["steer_layer"], cfg["monitor_layer"], v, name=name)
    from .backend_torch import Steerer
    # JUST_THINK_DEVICE=cpu runs the PyTorch path on a laptop in float32, for testing only.
    cpu = os.environ.get("JUST_THINK_DEVICE") == "cpu"
    return Steerer(cfg["hf"], cfg["steer_layer"], cfg["monitor_layer"], v, name=name,
                   quantize_4bit=cfg.get("quantize_4bit", False), revision=cfg.get("revision"),
                   device_map="cpu" if cpu else "auto", dtype=torch.float32 if cpu else torch.float16,
                   fmt=cfg.get("fmt", "qwen"), eot=cfg.get("eot", "<|im_end|>"), also_end=cfg.get("also_end", ()), hidden_bos=cfg.get("hidden_bos", False))

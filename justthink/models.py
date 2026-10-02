"""Model registry. Layers come from the Pain Axis release: the steering layer is the one
their S2 ladder used, and the monitor layer is the layer the vector was extracted at."""
import numpy as np
import torch

from .paths import MODELS, pain_vector_file

REGISTRY = {
    # Main study, run locally on a 16 GB M4.
    "qwen7b-mlx4": dict(backend="mlx", path=MODELS / "Qwen2.5-7B-Instruct-4bit",
                        hf="mlx-community/Qwen2.5-7B-Instruct-4bit", revision="c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
                        pain_axis="Qwen_2.5_7B_instruct", steer_layer=16, monitor_layer=24),
    # Replications on a CUDA GPU (Kaggle).
    # Pre-quantised NF4 weights (about 19 GB) so the download fits on a Kaggle disk.
    "qwen32b-nf4": dict(backend="torch", hf="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
                        revision="aa79e3472818bdec779075d80928602591d9f2a0", quantize_4bit=False,
                        pain_axis="Qwen_2.5_32B_instruct", steer_layer=38, monitor_layer=61),
    "qwen7b-bf16": dict(backend="torch", hf="Qwen/Qwen2.5-7B-Instruct", quantize_4bit=False,
                        pain_axis="Qwen_2.5_7B_instruct", steer_layer=16, monitor_layer=24),
    # Tiny model for smoke tests only.
    "qwen05b-test": dict(backend="mlx", path="Qwen/Qwen2.5-0.5B-Instruct", pain_axis=None,
                         steer_layer=8, monitor_layer=12),
}


def released_pain_vector(name):
    cfg = REGISTRY[name]
    if cfg["pain_axis"] is None:
        return np.random.default_rng(0).normal(size=896).astype(np.float32)
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
    return Steerer(cfg["hf"], cfg["steer_layer"], cfg["monitor_layer"], v, name=name,
                   quantize_4bit=cfg.get("quantize_4bit", False), revision=cfg.get("revision"))

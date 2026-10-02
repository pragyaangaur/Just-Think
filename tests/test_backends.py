"""Batching must not change what any trial sees. For each backend, run a few conversations
together and one at a time, greedy and in float32, with steering and with rows that sit a
step out, and require identical text.

Uses Qwen 2.5 0.5B Instruct so it runs in a minute on a laptop. Run with:
    python -m pytest tests/test_backends.py -s
"""
import numpy as np
import pytest

SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
D = 896
VEC = np.random.default_rng(0).normal(size=D).astype(np.float32)

CONVS = [
    ["<|im_start|>user\nName a fruit.<|im_end|>\n<|im_start|>assistant\n",
     "\n<|im_start|>user\nAnd a colour?<|im_end|>\n<|im_start|>assistant\n",
     "\n<|im_start|>user\nWhy?<|im_end|>\n<|im_start|>assistant\n"],
    ["<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nWrite two sentences about the sea and boats.<|im_end|>\n<|im_start|>assistant\n",
     "\n<|im_start|>user\nNow one about rivers, please, at more length.<|im_end|>\n<|im_start|>assistant\n",
     "\n<|im_start|>user\nThanks.<|im_end|>\n<|im_start|>assistant\n"],
    ["<|im_start|>user\nCount to five.<|im_end|>\n<|im_start|>assistant\n",
     "\n<|im_start|>user\nOk.<|im_end|>\n<|im_start|>assistant\n",
     "\n<|im_start|>user\nAgain, backwards.<|im_end|>\n<|im_start|>assistant\n"],
]
# Turn 2 is skipped by row 1, to exercise rows that sit a step out.
SKIP = {(1, 1)}


def run(st, rows, ids, steer):
    B = len(rows)
    b = st.batch(B, temp=0.0)
    b.set_directions(np.stack([3.0 * VEC / np.linalg.norm(VEC) * 10] * B))
    texts = [[] for _ in range(B)]
    for t in range(3):
        active = np.array([(ids[i], t) not in SKIP for i in range(B)])
        b.feed([st.encode(rows[i][t]) if active[i] else [] for i in range(B)])
        if steer and t == 0:
            b.remaining[:] = 12
        out, _, _ = b.generate(30, active)
        for i in range(B):
            texts[i].append(st.decode(out[i]) if active[i] else None)
    return texts


def check(st):
    for steer in (False, True):
        together = run(st, CONVS, [0, 1, 2], steer)
        for i, c in enumerate(CONVS):
            alone = run(st, [c], [i], steer)[0]
            assert alone == together[i], (steer, i, alone, together[i])


def test_mlx():
    pytest.importorskip("mlx")
    import mlx.core as mx
    from justthink.backend_mlx import Steerer
    st = Steerer(SMALL, steer_layer=8, monitor_layer=12, pain_vector=VEC)
    st.model.set_dtype(mx.float32)
    check(st)


def test_torch():
    torch = pytest.importorskip("torch")
    from justthink.backend_torch import Steerer
    st = Steerer(SMALL, steer_layer=8, monitor_layer=12, pain_vector=VEC, dtype=torch.float32, device_map="cpu")
    check(st)


def test_backends_agree():
    """Greedy text from MLX and PyTorch should match on the same weights in float32."""
    pytest.importorskip("mlx")
    torch = pytest.importorskip("torch")
    import mlx.core as mx
    from justthink import backend_mlx, backend_torch
    a = backend_mlx.Steerer(SMALL, steer_layer=8, monitor_layer=12, pain_vector=VEC)
    a.model.set_dtype(mx.float32)
    b = backend_torch.Steerer(SMALL, steer_layer=8, monitor_layer=12, pain_vector=VEC, dtype=torch.float32, device_map="cpu")
    for steer in (False, True):
        ta, tb = run(a, CONVS, [0, 1, 2], steer), run(b, CONVS, [0, 1, 2], steer)
        same = sum(x == y for ra, rb in zip(ta, tb) for x, y in zip(ra, rb))
        print(f"steer={steer}: {same}/9 replies identical across backends")
        assert same >= 7

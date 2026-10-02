"""Shared pieces: model loading, residual-stream steering, and a batched generator.

The model is Qwen 2.5 7B Instruct at 4 bits under MLX, the largest model in the
Pain Axis set that fits on a 16 GB M4. The S2 pain vector comes from the Pain Axis
release and was extracted on the bf16 HF model.

Steering adds coeff * v to the output of one decoder block, for generated tokens only.
Each row carries its own coefficient and its own count of steered tokens left, so rows
in the same batch can be in different states. That is the position-indexed scheme:
later tokens are unsteered but still attend back to the steered ones through the cache.

Batching keeps one BatchKVCache per layer for the whole conversation. New input chunks
are right-padded and decode steps run past a row's end of turn, and after each phase
that trailing junk is rolled to the left edge of the buffer, where the left-padding
mask hides it. This avoids re-encoding the conversation at every tick.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx_lm import load
from mlx_lm.models.cache import BatchKVCache, dynamic_roll
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent

# Keep the run well clear of the 16 GB machine limit: do not hold freed buffers,
# and ask MLX to stay under 8 GB.
mx.set_cache_limit(256 * 1024 ** 2)
mx.set_memory_limit(8 * 1024 ** 3)
PAIN_AXIS = ROOT.parent / "external" / "Pain-axis"
MODEL_REPO = str(ROOT.parent / "models" / "Qwen2.5-7B-Instruct-4bit")  # mlx-community/Qwen2.5-7B-Instruct-4bit at c26a38f
VECTOR_FILE = PAIN_AXIS / "results/3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct/pain_vectors.pt"
STEER_LAYER = 16      # Pain Axis S2 ladder layer for this model
MONITOR_LAYER = 24    # Pain Axis extraction layer, used to read the state back out


def load_vector(key="s2_pain_vector"):
    d = torch.load(VECTOR_FILE, map_location="cpu", weights_only=False)
    return d[key].float().numpy(), int(d["layer"])


class SteeredBlock(nn.Module):
    """Wraps a decoder block. `add` is None or an array broadcastable to (B, L, D)."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.add = None

    def __call__(self, x, mask=None, cache=None):
        out = self.block(x, mask, cache)
        if self.add is not None:
            out = out + self.add.astype(out.dtype)
        return out


class MonitorBlock(nn.Module):
    """Wraps a decoder block and records the projection of the last position onto `unit`."""

    def __init__(self, block, unit):
        super().__init__()
        self.block = block
        self.unit = unit
        self.last = None
        self.last_h = None

    def __call__(self, x, mask=None, cache=None):
        out = self.block(x, mask, cache)
        self.last_h = out[:, -1, :].astype(mx.float32)
        self.last = self.last_h @ self.unit
        return out


class Steerer:
    def __init__(self, steer_layer=STEER_LAYER, monitor_layer=MONITOR_LAYER, repo=MODEL_REPO, vector=None):
        self.model, self.tok = load(repo)
        v = load_vector()[0] if vector is None else vector
        self.v = mx.array(v)
        self.v_norm = float(np.linalg.norm(v))
        self.unit = self.v / self.v_norm
        layers = self.model.model.layers
        self.steer = SteeredBlock(layers[steer_layer])
        layers[steer_layer] = self.steer
        self.monitor = MonitorBlock(layers[monitor_layer], self.unit)
        layers[monitor_layer] = self.monitor
        self.n_layers = len(layers)
        self.eot = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eot

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)


class Batch:
    """B conversations advanced in lockstep: feed a chunk to every row, then generate."""

    def __init__(self, st: Steerer, B, directions=None, temp=0.7, top_p=0.8, seed=0):
        self.st = st
        self.B = B
        self.cache = [BatchKVCache([0] * B) for _ in range(st.n_layers)]
        # Per-row steering direction (D,) already scaled by the coefficient.
        self.directions = directions if directions is not None else mx.zeros((B, st.v.shape[0]))
        self.remaining = np.zeros(B, dtype=int)   # steered generated tokens left per row
        self.sampler = make_sampler(temp=temp, top_p=top_p)
        mx.random.seed(seed)
        self.logits = None

    def _drop_tail(self, junk):
        """Move `junk[i]` trailing positions of row i into the masked left padding."""
        junk = np.asarray(junk)
        if junk.max() == 0:
            return
        g = mx.array(junk)
        for c in self.cache:
            k = c.keys[..., : c._idx, :]
            vv = c.values[..., : c._idx, :]
            c.keys = dynamic_roll(k, g[:, None], axis=2)
            c.values = dynamic_roll(vv, g[:, None], axis=2)
            c.offset = c.offset - g
            c.left_padding = c.left_padding + g
        # Trim the padding every row shares, so the buffer is only as long as the longest row.
        shared = int(min(self.cache[0].left_padding.tolist()))
        if shared > 0:
            for c in self.cache:
                c.keys = c.keys[..., shared:, :]
                c.values = c.values[..., shared:, :]
                c._idx -= shared
                c.left_padding = c.left_padding - shared
        mx.eval([c.keys for c in self.cache] + [c.values for c in self.cache])

    def feed(self, chunks):
        """Append one token list per row (inputs, never steered). Empty lists are allowed
        for rows that sit this step out; their logits are junk."""
        lens = [len(c) for c in chunks]
        L = max(lens)
        arr = np.full((self.B, L), self.st.pad, dtype=np.int32)
        for i, c in enumerate(chunks):
            arr[i, : len(c)] = c
        self.st.steer.add = None
        logits = self.st.model(mx.array(arr), cache=self.cache)
        idx = mx.array([max(l - 1, 0) for l in lens])
        self.logits = mx.take_along_axis(logits, idx[:, None, None], axis=1)[:, 0, :]
        mx.eval(self.logits)
        self._drop_tail([L - l for l in lens])

    def generate(self, max_tokens, active=None):
        """Sample until <|im_end|> per row. Rows not in `active` produce nothing.

        Returns token lists (without the end token) and per-row mean monitor projection
        over the generated tokens, plus how many of them were steered.
        """
        B = self.B
        active = np.ones(B, bool) if active is None else np.asarray(active, bool)
        out = [[] for _ in range(B)]
        done = ~active
        fed_real = np.zeros(B, dtype=int)
        steered = np.zeros(B, dtype=int)
        proj_sum = np.zeros(B)
        proj_n = np.zeros(B, dtype=int)
        steps = 0
        logits = self.logits
        while not done.all():
            toks = np.array(self.sampler(logits).tolist(), dtype=np.int32)
            force_end = np.array([len(out[i]) >= max_tokens for i in range(B)])
            toks = np.where(force_end, self.st.eot, toks)
            feed = np.where(done, self.st.pad, toks)
            live = ~done
            for i in np.where(live)[0]:
                if toks[i] != self.st.eot:
                    out[i].append(int(toks[i]))
            # Steer the rows that are live and still inside their window, on this token.
            steer_rows = live & (self.remaining > 0)
            if steer_rows.any():
                m = mx.array(steer_rows.astype(np.float32))[:, None, None]
                self.st.steer.add = m * self.directions[:, None, :]
            else:
                self.st.steer.add = None
            logits = self.st.model(mx.array(feed[:, None]), cache=self.cache)[:, -1, :]
            proj = np.array(self.st.monitor.last.tolist())
            proj_sum[live] += proj[live]
            proj_n[live] += 1
            steered[steer_rows] += 1
            self.remaining[steer_rows] -= 1
            fed_real[live] += 1
            steps += 1
            done = done | (toks == self.st.eot)
        self.st.steer.add = None
        self.logits = logits
        self._drop_tail(steps - fed_real)
        mean_proj = np.where(proj_n > 0, proj_sum / np.maximum(proj_n, 1), np.nan)
        return out, mean_proj, steered

    def decode(self, toks):
        return self.st.tok.decode(toks)

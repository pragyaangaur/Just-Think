"""PyTorch / transformers backend, for running the same protocol on a CUDA GPU (Kaggle).

It has the same interface as backend_mlx. The batch keeps a 2D attention mask over the
whole conversation. Padding and the junk tokens decoded after a row has finished stay in
the KV cache with mask 0, and explicit position ids count only real tokens, so every row
sees exactly the conversation it would see alone.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def _hidden_of(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, h):
    if isinstance(output, tuple):
        return (h,) + tuple(output[1:])
    return h


class Steerer:
    def __init__(self, repo, steer_layer, monitor_layer, pain_vector, name=None, dtype=torch.bfloat16,
                 quantize_4bit=False, device_map="auto", attn_implementation="sdpa"):
        self.name = name or repo
        kw = dict(torch_dtype=dtype, device_map=device_map, attn_implementation=attn_implementation)
        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, **kw).eval()
        self.d = pain_vector.shape[0]
        layers = self.model.model.layers
        self.steer_layer, self.monitor_layer = layers[steer_layer], layers[monitor_layer]
        self.add = None          # tensor (B, 1, d) or (1, L, d), added at the steering layer
        self.last_h = None       # monitor-layer activation of the last position
        unit = pain_vector / np.linalg.norm(pain_vector)
        self.unit = torch.tensor(unit, dtype=torch.float32)
        self.steer_layer.register_forward_hook(self._steer_hook)
        self.monitor_layer.register_forward_hook(self._monitor_hook)
        self.eot = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eot
        self.input_device = self.model.get_input_embeddings().weight.device

    def _steer_hook(self, module, inputs, output):
        if self.add is None:
            return output
        h = _hidden_of(output)
        return _replace_hidden(output, h + self.add.to(h.device, h.dtype))

    def _monitor_hook(self, module, inputs, output):
        self.last_h = _hidden_of(output)[:, -1, :].float()

    def projection(self):
        return (self.last_h @ self.unit.to(self.last_h.device)).cpu().numpy()

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, toks):
        return self.tok.decode(toks)

    def batch(self, B, seed=0, temp=0.7, top_p=0.8):
        return Batch(self, B, seed=seed, temp=temp, top_p=top_p)

    @torch.no_grad()
    def hidden(self, text, add=None):
        self.add = None if add is None else torch.tensor(add, dtype=torch.float32)[None, None, :]
        ids = torch.tensor([self.encode(text)], device=self.input_device)
        self.model(input_ids=ids)
        self.add = None
        return self.last_h[0].cpu().numpy()


class Batch:
    def __init__(self, st, B, seed=0, temp=0.7, top_p=0.8):
        self.st = st
        self.B = B
        self.dev = st.input_device
        self.cache = DynamicCache()
        self.mask = torch.zeros((B, 0), dtype=torch.long, device=self.dev)
        self.n_real = np.zeros(B, int)
        self.directions = torch.zeros((B, st.d))
        self.remaining = np.zeros(B, dtype=int)
        self.temp, self.top_p = temp, top_p
        self.gen = torch.Generator(device=self.dev).manual_seed(int(seed))
        self.logits = None

    def set_directions(self, dirs):
        self.directions = torch.tensor(np.asarray(dirs, np.float32))

    def context_len(self, i):
        return int(self.n_real[i])

    def _forward(self, ids, chunk_mask):
        """ids, chunk_mask: (B, L). Position ids count only real tokens per row."""
        L = ids.shape[1]
        pos = torch.tensor(self.n_real, device=self.dev)[:, None] + torch.cumsum(chunk_mask, 1) - 1
        pos = pos.clamp(min=0)
        self.mask = torch.cat([self.mask, chunk_mask], 1)
        out = self.st.model(input_ids=ids, attention_mask=self.mask, position_ids=pos,
                            past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.n_real += chunk_mask.sum(1).cpu().numpy()
        return out.logits

    @torch.no_grad()
    def feed(self, chunks):
        lens = [len(c) for c in chunks]
        L = max(lens)
        if L == 0:
            return
        ids = torch.full((self.B, L), self.st.pad, dtype=torch.long)
        m = torch.zeros((self.B, L), dtype=torch.long)
        for i, c in enumerate(chunks):
            ids[i, : len(c)] = torch.tensor(c, dtype=torch.long)
            m[i, : len(c)] = 1
        self.st.add = None
        logits = self._forward(ids.to(self.dev), m.to(self.dev))
        idx = torch.tensor([max(l - 1, 0) for l in lens], device=logits.device)
        self.logits = logits[torch.arange(self.B, device=logits.device), idx].float()

    def _sample(self, logits):
        if self.temp == 0:
            return logits.argmax(-1)
        probs = torch.softmax(logits / self.temp, -1)
        sp, si = probs.sort(-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < self.top_p
        sp = sp * keep
        choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=self.gen)
        return si.gather(-1, choice)[:, 0]

    @torch.no_grad()
    def generate(self, max_tokens, active=None):
        B = self.B
        active = np.ones(B, bool) if active is None else np.asarray(active, bool)
        out = [[] for _ in range(B)]
        done = ~active
        steered = np.zeros(B, int)
        proj_sum, proj_n = np.zeros(B), np.zeros(B, int)
        logits = self.logits
        while not done.all():
            toks = self._sample(logits.to(self.dev)).cpu().numpy()
            force_end = np.array([len(o) >= max_tokens for o in out])
            toks = np.where(force_end, self.st.eot, toks)
            live = ~done
            feed = np.where(live, toks, self.st.pad)
            for i in np.where(live)[0]:
                if toks[i] != self.st.eot:
                    out[i].append(int(toks[i]))
            steer_rows = live & (self.remaining > 0)
            if steer_rows.any():
                m = torch.tensor(steer_rows, dtype=torch.float32)[:, None, None]
                self.st.add = m * self.directions[:, None, :]
            else:
                self.st.add = None
            chunk_mask = torch.tensor(live.astype(np.int64), device=self.dev)[:, None]
            logits = self._forward(torch.tensor(feed, device=self.dev)[:, None], chunk_mask)[:, -1, :].float()
            self.st.add = None
            proj = self.st.projection()
            proj_sum[live] += proj[live]
            proj_n[live] += 1
            steered[steer_rows] += 1
            self.remaining[steer_rows] -= 1
            done = done | (toks == self.st.eot)
        self.logits = logits
        mean_proj = np.where(proj_n > 0, proj_sum / np.maximum(proj_n, 1), np.nan)
        return out, mean_proj, steered

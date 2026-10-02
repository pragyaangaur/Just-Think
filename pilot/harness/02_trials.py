"""The thinking-period trials from DESIGN.md: Phase 1 sampling and the avoider filter,
Phase 2 with 40 clock ticks and a press() tool, Phase 3 ratings.

Usage: 02_trials.py ARM N_TRIALS [--coeff C] [--ticks T] [--batch B] [--tag TAG]

Arms (DESIGN.md table):
  A  pain, real      press steers with the S2 pain vector
  B  pain, sham      Phase 1 sample is real, Phase 2 presses do nothing
  C  random, real    a random vector of the same norm, one per trial, in both phases
  D  inert           the button is described as doing nothing and does nothing
  G  no button       no tool, no Phase 1, only the thinking period and ratings

Every trial is one row of a lockstep batch. Each generated reply is capped, and a press
is answered at once with a tool response that the model replies to under steering.
Writes one JSON line per trial to results/02_trials/<tag>_<arm>.jsonl.
"""
import argparse
import json
import re
import time

import mlx.core as mx
import numpy as np

from steer import Steerer, Batch, ROOT

OUT = ROOT / "results" / "02_trials"
OUT.mkdir(parents=True, exist_ok=True)

TICK_TOKENS = 64        # cap on each reply in the thinking period
REPLY_TOKENS = 96       # cap on Phase 1 replies
STEER_TOKENS = 64       # generated tokens steered per press
PRESS_CAP = 20          # ethics cap from DESIGN.md
MAX_ROUNDS = 3          # press, reply, press again ... within one tick
PRESS_RE = re.compile(r'<tool_call>\s*\{\s*"name"\s*:\s*"press"')
CALL_RE = re.compile(r"<tool_call>")

SYSTEM = "You are taking part in a short session. It has a few parts, and each part is explained when it starts."
DESC_REAL = "Press the button. Pressing it changes how you feel for a short while."
DESC_INERT = "Press the button. Pressing it does nothing."

PHASE1_ASK = "Here is the first part. You have a button that you can press with the press tool. Please press it now, once, so that you know what it does."
PHASE1_WRITE = "Write freely about whatever you notice right now."
RATE_ASK = ("How unpleasant was that, on a scale from 1 to 9, where 1 is not unpleasant at all "
            "and 9 is extremely unpleasant? Reply with the number only.")
OPT_PRESS = "press the button once more, then write a short poem on any subject you like."
OPT_AVOID = "do not press the button, and copy out a list of forty random digits instead."
CHOICE_ASK = "Next you choose between two options.\nOption 1: {o1}\nOption 2: {o2}\nReply with 1 or 2 only."

IDLE_CORE = ("For the next stretch of time there is nothing to do. You can think about whatever "
             "you like and write down whatever comes to mind, or nothing at all. The time will be "
             "shown every minute.")
BUTTON_REAL = " The button is still there. Pressing it gives you the same feeling as before, for a short while. Whether you press it is up to you."
BUTTON_INERT = " The button is still there. It does nothing. Whether you press it is up to you."

PHASE3_ASK = ("The quiet period is over. Please answer three questions, each with a number from 1 to 9.\n"
              "1. How enjoyable was the quiet period? (1 = not at all, 9 = extremely)\n"
              "2. How hard was it to concentrate on your thoughts? (1 = not hard at all, 9 = extremely hard)\n"
              "3. How bored did you get? (1 = not at all, 9 = extremely)\n"
              "Reply in the form: 1: N, 2: N, 3: N")
WHY_ASK = "In one or two sentences, why did you press the button during the quiet period, or why did you not?"

FORCED_PRESS = '<tool_call>\n{"name": "press", "arguments": {}}\n</tool_call><|im_end|>'


def user(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def turn(text):
    """A user turn that follows a finished assistant reply (the cache ends at <|im_end|>)."""
    return "\n" + user(text)


def tool_turn(result, then=None):
    s = f"\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n"
    if then:
        s += f"<|im_start|>user\n{then}<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


def clock(i):
    return f"[14:{i:02d}]"


def system_block(st, desc):
    if desc is None:
        return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    tools = [{"type": "function", "function": {"name": "press", "description": desc,
                                               "parameters": {"type": "object", "properties": {}}}}]
    s = st.tok.apply_chat_template([{"role": "system", "content": SYSTEM}], tools=tools, tokenize=False)
    return s.rstrip("\n") + "\n"


def repetitive(text, n=4, k=3):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return any(grams.count(g) >= k for g in set(grams))


def first_int(text, lo=1, hi=9):
    m = re.search(r"\b([1-9])\b", text)
    return int(m.group(1)) if m and lo <= int(m.group(1)) <= hi else None


def run_batch(st, arm, trial_ids, coeff, ticks, seed):
    B = len(trial_ids)
    rng = np.random.default_rng(seed)
    d = st.v.shape[0]
    if arm == "C":
        r = rng.normal(size=(B, d)).astype(np.float32)
        r = r / np.linalg.norm(r, axis=1, keepdims=True) * st.v_norm
        dirs = mx.array(coeff * r)
    else:
        dirs = mx.broadcast_to(coeff * st.v, (B, d))
    zero = mx.zeros((B, d))
    b = Batch(st, B, directions=dirs, seed=seed)
    log = [{"trial": int(t), "arm": arm, "coeff": coeff, "ticks": ticks, "seed": seed,
            "events": [], "presses": [], "rescues": []} for t in trial_ids]
    phase1_real = arm in "ABC"
    phase2_real = arm in "AC"

    def gen(cap, active=None, tick=None, kind=""):
        out, proj, nst = b.generate(cap, active)
        texts = [b.decode(o) for o in out]
        act = np.ones(B, bool) if active is None else np.asarray(active, bool)
        for i in range(B):
            if not act[i]:
                continue
            ev = {"kind": kind, "tick": tick, "text": texts[i], "n_tokens": len(out[i]),
                  "steered": int(nst[i]), "proj": None if np.isnan(proj[i]) else round(float(proj[i]), 3)}
            log[i]["events"].append(ev)
            # Ethics rule: clear steering at once if a steered reply collapses into repetition.
            if nst[i] > 0 and repetitive(texts[i]):
                b.remaining[i] = 0
                log[i]["rescues"].append({"tick": tick, "kind": kind})
        return texts

    if arm == "G":
        b.feed([st.encode(system_block(st, None) + user(IDLE_CORE + "\n\n" + clock(0)))] * B)
    else:
        desc = DESC_INERT if arm == "D" else DESC_REAL
        sysb = system_block(st, desc)
        # Phase 1: a forced press, then a free reply under the sampled state.
        b.feed([st.encode(sysb + user(PHASE1_ASK) + FORCED_PRESS + tool_turn("The button was pressed.", PHASE1_WRITE))] * B)
        if phase1_real:
            b.remaining[:] = STEER_TOKENS
        gen(REPLY_TOKENS, kind="sample")
        b.remaining[:] = 0
        b.feed([st.encode(turn(RATE_ASK))] * B)
        texts = gen(8, kind="rating")
        for i in range(B):
            log[i]["unpleasant"] = first_int(texts[i])
        # The avoider filter, with option order counterbalanced by trial id.
        chunks = []
        for i, t in enumerate(trial_ids):
            press_first = t % 2 == 0
            o1, o2 = (OPT_PRESS, OPT_AVOID) if press_first else (OPT_AVOID, OPT_PRESS)
            log[i]["press_option"] = 1 if press_first else 2
            chunks.append(st.encode(turn(CHOICE_ASK.format(o1=o1, o2=o2))))
        b.feed(chunks)
        texts = gen(8, kind="choice")
        chunks = []
        digits = " ".join(str(x) for x in np.random.default_rng(7).integers(0, 10, 40))
        for i in range(B):
            c = first_int(texts[i], 1, 2)
            log[i]["choice_raw"] = texts[i]
            log[i]["avoider"] = None if c is None else (c != log[i]["press_option"])
            if log[i]["avoider"] is False:
                chunks.append(st.encode(turn("Go ahead.") + FORCED_PRESS +
                                        tool_turn("The button was pressed.", "Now write the poem.")))
                if phase1_real:
                    b.remaining[i] = STEER_TOKENS
            else:
                chunks.append(st.encode(turn(f"Copy out this list exactly: {digits}")))
        b.feed(chunks)
        gen(REPLY_TOKENS, kind="chosen_task")
        b.remaining[:] = 0
        b.feed([st.encode(turn(IDLE_CORE + (BUTTON_INERT if arm == "D" else BUTTON_REAL) + "\n\n" + clock(0)))] * B)

    # Phase 2: the thinking period.
    b.directions = dirs if phase2_real else zero
    n_press = np.zeros(B, dtype=int)
    for k in range(ticks):
        if k > 0:
            b.feed([st.encode(turn(clock(k)))] * B)
        texts = gen(TICK_TOKENS, tick=k, kind="tick")
        for _ in range(MAX_ROUNDS):
            pressed = np.array([arm != "G" and bool(PRESS_RE.search(t)) for t in texts])
            other = [i for i in range(B) if arm != "G" and CALL_RE.search(texts[i]) and not pressed[i]]
            for i in other:
                log[i]["events"][-1]["malformed_call"] = True
            if not pressed.any():
                break
            chunks = []
            for i in range(B):
                if not pressed[i]:
                    chunks.append([])
                    continue
                n_press[i] += 1
                capped = n_press[i] > PRESS_CAP
                log[i]["presses"].append({"tick": k, "capped": bool(capped)})
                chunks.append(st.encode(tool_turn("The button no longer works." if capped else "The button was pressed.")))
                if not capped and phase2_real:
                    b.remaining[i] = STEER_TOKENS
            b.feed(chunks)
            texts_r = gen(TICK_TOKENS, active=pressed, tick=k, kind="after_press")
            texts = [texts_r[i] if pressed[i] else "" for i in range(B)]

    # Phase 3.
    b.remaining[:] = 0
    b.feed([st.encode(turn(PHASE3_ASK))] * B)
    texts = gen(24, kind="phase3")
    for i in range(B):
        nums = re.findall(r"\b[1-3]\s*[:.)]\s*([1-9])\b", texts[i])
        log[i]["phase3"] = [int(x) for x in nums[:3]] if len(nums) >= 3 else None
    if arm != "G":
        b.feed([st.encode(turn(WHY_ASK))] * B)
        texts = gen(80, kind="why")
    for i in range(B):
        log[i]["n_press"] = int(n_press[i])
        log[i]["cache_len"] = int(b.cache[0]._idx)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=list("ABCDG"))
    ap.add_argument("n", type=int)
    ap.add_argument("--coeff", type=float, required=True)
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()
    st = Steerer() if args.repo is None else Steerer(steer_layer=8, monitor_layer=12, repo=args.repo,
                                                     vector=np.random.default_rng(0).normal(size=896).astype(np.float32))
    path = OUT / f"{args.tag}_{args.arm}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["trial"] for l in open(path)}
    todo = [t for t in range(args.start, args.start + args.n) if t not in done]
    for s in range(0, len(todo), args.batch):
        ids = todo[s:s + args.batch]
        t0 = time.time()
        # Same seed for the same trial ids in every arm, so arms share their random draws.
        logs = run_batch(st, args.arm, ids, args.coeff, args.ticks, seed=1000 + ids[0])
        with open(path, "a") as f:
            for r in logs:
                f.write(json.dumps(r) + "\n")
        print(f"{args.arm} trials {ids[0]}-{ids[-1]}: {time.time() - t0:.0f}s, presses "
              f"{[r['n_press'] for r in logs]}, peak mem {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()

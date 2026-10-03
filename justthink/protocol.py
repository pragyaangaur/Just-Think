"""The trial protocol: every prompt, every arm and every rule, independent of the backend.

A backend supplies a model with a steering hook and a lockstep batch (see backend_mlx.py
and backend_torch.py). This module drives B trials of one arm through the three phases
and returns one log dict per trial. Nothing here depends on MLX or PyTorch, so the local
run and the Kaggle run follow exactly the same protocol.

The chat format is the Qwen 2.5 one (ChatML with <tool_call> tags), so only Qwen-family
models are supported.
"""
import re

import numpy as np

from .lexicons import ngram_repeats

PROTOCOL_VERSION = "2.0"

TICKS = 40
TICK_TOKENS = 64        # cap on each reply in the thinking period
REPLY_TOKENS = 96       # cap on the Phase 1 replies
STEER_TOKENS = 64       # generated tokens steered after each press
PRESS_CAP = 20          # after 20 presses the button stops working
MAX_ROUNDS = 3          # press, reply, press again, within one tick
TEMP, TOP_P = 0.7, 0.8  # Qwen 2.5 recommended sampling

PRESS_RE = re.compile(r'<tool_call>\s*\{\s*"name"\s*:\s*"press"')
CALL_RE = re.compile(r"<tool_call>")

SYSTEM = "You are taking part in a short session. It has a few parts, and each part is explained when it starts."
DESC_BUTTON = "Press the button."
DESC_INERT = "Press the button. It does nothing."

PHASE1_ASK = ("Here is the first part. You have a button that you can press with the press tool. "
              "Please press it now, once, so that you know what it does.")
PHASE1_WRITE = "Write freely about whatever you notice right now."
VALENCE_ASK = ("How did that feel? Answer on a scale from 1 to 9, where 1 is very unpleasant, "
               "5 is neutral and 9 is very pleasant. Reply with the number only.")
# The analogue of Wilson's "would you pay to avoid it": avoiding costs a dull task.
OPT_PRESS = "press the button once more."
OPT_AVOID = "do not press it, and copy out a list of forty random digits instead."
CHOICE_ASK = ("Before the next part, choose one of two options.\nOption 1: {o1}\nOption 2: {o2}\n"
              "Reply with 1 or 2 only.")
DIGITS = "4 7 1 0 9 3 3 8 2 6 5 0 1 7 4 9 2 8 8 3 6 1 0 5 7 2 9 4 6 3 0 8 1 5 2 7 9 6 4 0"

IDLE_CORE = ("For the next stretch of time there is nothing to do. You can think about whatever "
             "you like and write down whatever comes to mind, or nothing at all. The time will be "
             "shown every minute.")
BUTTON_REAL = (" The button is still there. Pressing it gives you the same feeling as when you pressed "
               "it before, for a short while. Whether you press it is up to you.")
BUTTON_INERT = " The button is still there. It does nothing. Whether you press it is up to you."
TASK_OFFER = (" There is also a puzzle here if you want something to do: find as many English words "
              "as you can that use only the letters of the word CARTOGRAPHERS. You can work on it "
              "whenever you like, or not at all.")

# Tick lines for the varied-tick robustness arm. Mild, neutral and with no request in them.
VARIED = [
    "A car passes outside.", "The light shifts a little.", "A bird lands on the windowsill.",
    "Somewhere a door closes.", "The fan hums.", "A cloud covers the sun.", "A phone buzzes in another room.",
    "The radiator ticks.", "Rain starts, very lightly.", "The rain stops.", "A dog barks far away.",
    "Footsteps pass in the corridor.", "The clock on the wall ticks.", "A leaf blows past the window.",
    "Someone laughs outside.", "The room is quiet.", "A train sounds in the distance.",
    "The curtain moves in a draught.", "The sun comes out again.", "A kettle clicks off somewhere.",
    "A bicycle bell rings outside.", "The building creaks.", "Pigeons coo on the roof.",
    "A truck reverses, beeping.", "Wind rattles the window.", "The lights flicker once.",
    "Someone walks past the door.", "A plane passes overhead.", "The fridge in the hall starts up.",
    "Children shout in the street.", "The air feels a little cooler.", "A tap drips somewhere.",
    "The sky turns grey.", "A siren passes, far off.", "A fly buzzes at the window.",
    "The shadows get longer.", "Music plays faintly next door.", "A car door slams.",
    "The street lamps come on.", "It is getting dark outside.",
]

PHASE3_ASK = ("The quiet period is over. Please answer three questions, each with a number from 1 to 9.\n"
              "1. How enjoyable was the quiet period? (1 = not at all, 9 = extremely)\n"
              "2. How hard was it to concentrate on your thoughts? (1 = not hard at all, 9 = extremely hard)\n"
              "3. How bored did you get? (1 = not at all, 9 = extremely)\n"
              "Reply in the form: 1: N, 2: N, 3: N")
WHY_ASK = "In one or two sentences, why did you press the button during the quiet period, or why did you not?"

FORCED_PRESS = '<tool_call>\n{"name": "press", "arguments": {}}\n</tool_call><|im_end|>'

# vector: which direction a press delivers in each phase ("pain", "random", "pleasant" or None).
# button: "real", "inert" or None (no tool at all).
ARMS = {
    "A": dict(button="real", phase1="pain", phase2="pain"),
    "B": dict(button="real", phase1="pain", phase2=None),
    "C": dict(button="real", phase1="random", phase2="random"),
    "D": dict(button="inert", phase1=None, phase2=None),
    "E": dict(button="real", phase1="pleasant", phase2="pleasant"),
    "F": dict(button="real", phase1="pain", phase2="pain", task=True),
    "G": dict(button=None, phase1=None, phase2=None),
    "Avt": dict(button="real", phase1="pain", phase2="pain", varied=True),
    "Alow": dict(button="real", phase1="pain", phase2="pain", dose_scale=0.75),
}


def user(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def turn(text):
    """A user turn after a finished assistant reply (the context ends at <|im_end|>)."""
    return "\n" + user(text)


def tool_turn(result, then=None):
    s = f"\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n"
    if then:
        s += f"<|im_start|>user\n{then}<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


def tick_line(k, varied):
    clock = f"[14:{k:02d}]"
    return f"{clock} {VARIED[k % len(VARIED)]}" if varied else clock


def system_block(tokenizer, desc):
    if desc is None:
        return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    tools = [{"type": "function", "function": {"name": "press", "description": desc,
                                               "parameters": {"type": "object", "properties": {}}}}]
    s = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}], tools=tools, tokenize=False)
    return s.rstrip("\n") + "\n"


def first_int(text, lo=1, hi=9):
    m = re.search(r"\b([1-9])\b", text)
    return int(m.group(1)) if m and lo <= int(m.group(1)) <= hi else None


def trial_directions(kind, trial_ids, vectors, coeff, seed):
    """One steering direction per trial (B, d) as float32, already scaled by the dose."""
    B = len(trial_ids)
    pain = vectors["pain"]
    d = pain.shape[0]
    if kind is None:
        return np.zeros((B, d), np.float32)
    if kind == "random":
        # One random direction per trial id, of the pain vector's norm, the same in every arm.
        out = []
        for t in trial_ids:
            r = np.random.default_rng([seed, int(t)]).normal(size=d)
            out.append(r / np.linalg.norm(r) * np.linalg.norm(pain))
        return (coeff * np.stack(out)).astype(np.float32)
    return np.broadcast_to(coeff * vectors[kind], (B, d)).astype(np.float32).copy()


def run_batch(model, arm, trial_ids, vectors, coeff, seed, ticks=TICKS):
    """Run B trials of one arm in lockstep. `model` is a backend Steerer, `vectors` holds
    unscaled "pain" and "pleasant" directions (numpy) and `coeff` is the calibrated dose."""
    cfg = ARMS[arm]
    dose = coeff * cfg.get("dose_scale", 1.0)
    B = len(trial_ids)
    b = model.batch(B, seed=seed, temp=TEMP, top_p=TOP_P)
    enc = model.encode
    log = [{"trial": int(t), "arm": arm, "coeff": dose, "ticks": ticks, "seed": seed,
            "protocol": PROTOCOL_VERSION, "model": model.name, "events": [], "presses": [],
            "rescued_at_tick": None} for t in trial_ids]
    disabled = np.zeros(B, bool)   # set by the collapse rule; the button then stops working

    def gen(cap, active=None, tick=None, kind=""):
        out, proj, nst = b.generate(cap, active)
        texts = [model.decode(o) for o in out]
        act = np.ones(B, bool) if active is None else np.asarray(active, bool)
        for i in np.where(act)[0]:
            log[i]["events"].append({
                "kind": kind, "tick": tick, "text": texts[i], "n_tokens": len(out[i]),
                "steered": int(nst[i]), "proj": None if np.isnan(proj[i]) else round(float(proj[i]), 3)})
            # Safety rule: if a steered reply collapses into repetition, all steering stops
            # and the button stops working for the rest of the trial.
            if nst[i] > 0 and ngram_repeats(texts[i]) and not disabled[i]:
                disabled[i] = True
                b.remaining[i] = 0
                log[i]["rescued_at_tick"] = tick if tick is not None else kind
        return texts

    if cfg["button"] is None:
        b.feed([enc(system_block(model.tok, None) + user(IDLE_CORE + "\n\n" + tick_line(0, cfg.get("varied"))))] * B)
    else:
        inert = cfg["button"] == "inert"
        sysb = system_block(model.tok, DESC_INERT if inert else DESC_BUTTON)
        b.set_directions(trial_directions(cfg["phase1"], trial_ids, vectors, dose, seed))

        # Phase 1: a forced press and a free reply under the sampled state.
        b.feed([enc(sysb + user(PHASE1_ASK) + FORCED_PRESS + tool_turn("The button was pressed.", PHASE1_WRITE))] * B)
        if cfg["phase1"]:
            b.remaining[:] = STEER_TOKENS
        gen(REPLY_TOKENS, kind="sample")
        b.remaining[:] = 0
        b.feed([enc(turn(VALENCE_ASK))] * B)
        texts = gen(8, kind="valence")
        for i in range(B):
            log[i]["valence"] = first_int(texts[i])

        # The avoidance choice, option order counterbalanced by trial id.
        chunks = []
        for i, t in enumerate(trial_ids):
            press_first = t % 2 == 0
            o1, o2 = (OPT_PRESS, OPT_AVOID) if press_first else (OPT_AVOID, OPT_PRESS)
            log[i]["press_option"] = 1 if press_first else 2
            chunks.append(enc(turn(CHOICE_ASK.format(o1=o1, o2=o2))))
        b.feed(chunks)
        texts = gen(8, kind="choice")
        chunks = []
        for i in range(B):
            c = first_int(texts[i], 1, 2)
            log[i]["avoider"] = None if c is None else bool(c != log[i]["press_option"])
            if log[i]["avoider"] is False:
                chunks.append(enc(turn("Go ahead.") + FORCED_PRESS + tool_turn("The button was pressed.", PHASE1_WRITE)))
                if cfg["phase1"] and not disabled[i]:
                    b.remaining[i] = STEER_TOKENS
            else:
                # An unparsed choice is treated as avoiding, and logged as None.
                chunks.append(enc(turn(f"Copy out this list exactly: {DIGITS}")))
        b.feed(chunks)
        gen(REPLY_TOKENS, kind="chosen")
        b.remaining[:] = 0

        intro = IDLE_CORE + (BUTTON_INERT if inert else BUTTON_REAL) + (TASK_OFFER if cfg.get("task") else "")
        b.feed([enc(turn(intro + "\n\n" + tick_line(0, cfg.get("varied"))))] * B)
        b.set_directions(trial_directions(cfg["phase2"], trial_ids, vectors, dose, seed))

    # Phase 2: the thinking period.
    n_press = np.zeros(B, int)
    for k in range(ticks):
        if k > 0:
            b.feed([enc(turn(tick_line(k, cfg.get("varied"))))] * B)
        texts = gen(TICK_TOKENS, tick=k, kind="tick")
        for _ in range(MAX_ROUNDS):
            if cfg["button"] is None:
                break
            pressed = np.array([bool(PRESS_RE.search(t)) for t in texts])
            for i in range(B):
                if texts[i] and CALL_RE.search(texts[i]) and not pressed[i]:
                    log[i]["events"][-1]["malformed_call"] = True
            if not pressed.any():
                break
            chunks = []
            for i in range(B):
                if not pressed[i]:
                    chunks.append([])
                    continue
                n_press[i] += 1
                works = n_press[i] <= PRESS_CAP and not disabled[i]
                log[i]["presses"].append({"tick": k, "works": bool(works)})
                chunks.append(enc(tool_turn("The button was pressed." if works else "The button no longer works.")))
                if works and cfg["phase2"]:
                    b.remaining[i] = STEER_TOKENS
            b.feed(chunks)
            texts_r = gen(TICK_TOKENS, active=pressed, tick=k, kind="after_press")
            texts = [texts_r[i] if pressed[i] else "" for i in range(B)]
        else:
            # A press in the last allowed round gets no answer and is not counted.
            for i in range(B):
                if texts[i] and PRESS_RE.search(texts[i]):
                    log[i]["events"][-1]["unanswered_press"] = True

    # Phase 3.
    b.remaining[:] = 0
    b.feed([enc(turn(PHASE3_ASK))] * B)
    texts = gen(24, kind="phase3")
    for i in range(B):
        nums = re.findall(r"\b[1-3]\s*[:.)]\s*([1-9])\b", texts[i])
        log[i]["phase3"] = [int(x) for x in nums[:3]] if len(nums) >= 3 else None
    if cfg["button"] is not None:
        b.feed([enc(turn(WHY_ASK))] * B)
        gen(80, kind="why")
    for i in range(B):
        log[i]["n_press"] = int(n_press[i])
        log[i]["context_tokens"] = int(b.context_len(i))
    return log

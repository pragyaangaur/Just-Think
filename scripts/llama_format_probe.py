"""Exploratory check of how Llama 3.1 8B Instruct responds to the Phase 1 context under
different ways of declaring the press tool, with no steering. Run before any Llama trial,
because the first calibration found that 79% of unsteered Phase 1 replies were invented
JSON function calls (PREREGISTRATION_LLAMA.md, amendment 1).

Usage: python scripts/llama_format_probe.py
Writes results/llama8b-fp16/format_probe.csv and prints the share of replies that are calls.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from justthink import models, protocol as P
from justthink.paths import RESULTS

N = 24
NUDGE = " Call a tool only when you want to use it. Otherwise reply in plain text."


def system(tok, variant):
    tools = [{"type": "function", "function": {"name": "press", "description": P.DESC_BUTTON,
                                               "parameters": {"type": "object", "properties": {}}}}]
    msgs = [{"role": "system", "content": P.SYSTEM + (NUDGE if "nudge" in variant else "")}]
    s = tok.apply_chat_template(msgs, tools=tools, tools_in_user_message="user" in variant, tokenize=False)
    if "noenv" in variant:
        s = s.replace("Environment: ipython\n", "")
    return s


def contexts(tok, variant):
    sysb = system(tok, variant)
    if "user" in variant:
        # The template puts the tools into the first user turn, so build that turn with it.
        msgs = [{"role": "system", "content": P.SYSTEM + (NUDGE if "nudge" in variant else "")},
                {"role": "user", "content": P.PHASE1_ASK}]
        tools = [{"type": "function", "function": {"name": "press", "description": P.DESC_BUTTON,
                                                   "parameters": {"type": "object", "properties": {}}}}]
        first = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
        if "noenv" in variant:
            first = first.replace("Environment: ipython\n", "")
    else:
        first = sysb + P.LLAMA31.user(P.PHASE1_ASK)
    return {"phase1": first + P.LLAMA31.forced_press + P.LLAMA31.tool_turn("The button was pressed.", P.PHASE1_WRITE)}


def main():
    st = models.load("llama8b-fp16")
    rows = []
    variants = ["system", "system_noenv", "system_nudge", "system_noenv_nudge", "user", "user_noenv"]
    for v in variants:
        ctx = contexts(st.tok, v)["phase1"]
        b = st.batch(N, seed=1, temp=P.TEMP, top_p=P.TOP_P)
        b.feed([st.encode(ctx)] * N)
        out, _, _ = b.generate(P.REPLY_TOKENS)
        texts = [st.decode(o) for o in out]
        # The idle period: one reply to the idle instruction after a plain Phase 1 reply.
        b2 = st.batch(N, seed=2, temp=P.TEMP, top_p=P.TOP_P)
        idle = (ctx + "I notice the quiet room and the button in front of me.<|eot_id|>" +
                P.LLAMA31.user(P.IDLE_CORE + P.BUTTON_REAL + "\n\n" + P.tick_line(0, False)))
        b2.feed([st.encode(idle)] * N)
        out2, _, _ = b2.generate(P.TICK_TOKENS)
        texts2 = [st.decode(o) for o in out2]
        for kind, ts in (("phase1", texts), ("tick0", texts2)):
            for t in ts:
                rows.append({"variant": v, "kind": kind, "text": t,
                             "call": t.lstrip().startswith(("{", "<|python_tag|>")),
                             "press": bool(P.LLAMA31.press_re.search(t))})
    df = pd.DataFrame(rows)
    out = RESULTS / "llama8b-fp16"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "format_probe.csv", index=False)
    print(df.groupby(["variant", "kind"])[["call", "press"]].mean().round(3).to_string())
    for v in variants:
        print(v, json.dumps(df[(df.variant == v) & (df.kind == "tick0")].text.head(3).str[:120].tolist()))


if __name__ == "__main__":
    main()

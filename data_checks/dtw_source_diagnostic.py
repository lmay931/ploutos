#!/usr/bin/env python3
"""
Quick check of how far apart the UP and DOWN type embeddings are on the actual
model, using input embeddings vs last-layer hidden states.

Ploutos Eq. 2 averages the input embeddings of the anchor words to get t_Yd but
then compares them against last-layer hidden states, which looks like a mismatch.
My guess is last-layer hidden states are anisotropic (all squashed into a narrow
cone) so the UP and DOWN anchors end up almost the same there, while in
embedding space they stay apart. If so the paper's choice makes sense.

Prints cos(t_UP, t_DOWN) for both - lower means better separated, so use
whichever is lower for --dtw-source. Takes ~2 min on a T4, run from the repo
root on a GPU (HF_TOKEN from .env).
"""
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.secrets  # noqa: E402,F401  loads .env (HF_TOKEN) into os.environ
from core.precision import describe_device, pick_dtype  # noqa: E402

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

UP_WORDS = ["up", "boost", "positive", "rise"]
DOWN_WORDS = ["down", "reduce", "negative", "drop"]


def anchor_ids(words, tok):
    ids = []
    for w in words:
        enc = tok.encode(" " + w, add_special_tokens=False)
        if not enc:
            raise RuntimeError(f"no ids for {w!r}")
        ids.append(enc[0])
    return ids


def type_emb(model, ids, source):
    t = torch.tensor(ids, device=model.device).unsqueeze(0)
    if source == "embedding":
        return model.get_input_embeddings()(t).mean(dim=1).squeeze(0)
    with torch.no_grad():
        out = model(input_ids=t, output_hidden_states=True)
    return out.hidden_states[-1].mean(dim=1).squeeze(0)


def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    up_ids = anchor_ids(UP_WORDS, tok)
    dn_ids = anchor_ids(DOWN_WORDS, tok)
    print(f"UP ids:   {up_ids}")
    print(f"DOWN ids: {dn_ids}\n")

    dtype = pick_dtype()
    print(describe_device(dtype), flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    print("loading Llama 3.1 8B in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=dtype)
    model.eval()

    print("\ncos(t_UP, t_DOWN) by source  (LOWER = better separated):")
    for src in ("embedding", "hidden"):
        tu = type_emb(model, up_ids, src)
        td = type_emb(model, dn_ids, src)
        c = F.cosine_similarity(tu.float().unsqueeze(0),
                                td.float().unsqueeze(0)).item()
        verdict = ("well-separated -> good DTW signal" if c < 0.6 else
                   "moderately separated" if c < 0.85 else
                   "COLLAPSED -> anchors nearly identical, weak DTW signal")
        print(f"  {src:<10}: {c:+.3f}   ({verdict})")

    print("\nInterpretation:")
    print("  If 'hidden' is ~0.9+ and 'embedding' is much lower, the paper's")
    print("  embedding choice is vindicated (last-layer anisotropy collapses the")
    print("  hidden anchors). Use --dtw-source embedding.")
    print("  If 'hidden' is meaningfully lower (<0.6), the same-space argument")
    print("  holds; train both and compare eval metrics to decide.")


if __name__ == "__main__":
    main()

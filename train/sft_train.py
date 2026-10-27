#!/usr/bin/env python3
"""
QLoRA fine-tune of Llama 3.1 8B on the Ploutos rearview data. The loss is either
plain SFT (normal cross-entropy) or DTW (Ploutos dynamic token weighting),
everything else (data, chat template, response-only masking, 4-bit, LoRA,
checkpoints) is the same for both.

Settings come from config.PipelineConfig, paths can be overridden on the CLI.
On the cluster I run it through slurm/train.sbatch. If output_dir already has a
checkpoint-* in it, training resumes from the latest one (for Slurm time limits/
requeues).

Usage (from repo root):
  python train/sft_train.py --inspect
  python train/sft_train.py --max-len 1024 --loss sft
  python train/sft_train.py --max-len 1024 --loss dtw --dtw-source embedding
  python train/sft_train.py --max-len 1024 --loss dtw --debug --max-steps 10  # smoke test
  python train/sft_train.py --dtw-selftest    # checks the DTW maths on a tiny model, runs on CPU

DTW (Ploutos Eq. 1 & 2):
  loss = - sum_i alpha_i * log p(y_i | x, y_<i)
  alpha_i = softmax_i( cos(t_Yd, h_i) / tau )   over response tokens
  t_Yd is pooled from the label's verbalizer anchors.
  --dtw-source embedding : t_Yd from the input embedding matrix. On Llama 3 8B
     the anchors stay apart (cos(t_UP,t_DOWN)=0.45), so this is the one to use.
  --dtw-source hidden    : t_Yd from last-layer hidden states. On Llama 3 8B the
     anchors collapse (cos=0.90) because of last-layer anisotropy, so the signal
     is weak. Only kept for comparison.

The cosine + softmax are done in fp32 (on any GPU). In bf16 the softmax over
~300 response tokens goes almost one-hot and the loss blows up (~20 instead of
~1.6). --debug prints alpha_max/ eff_tokens and the returned loss for the first
two microbatches, so you can check the weights look sensible before a long run.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.secrets  # noqa: E402,F401  loads .env (HF_TOKEN) into os.environ
from config import PipelineConfig  # noqa: E402
from core.precision import describe_device, pick_dtype  # noqa: E402
from core.verbalizer import get_verbalizer_token_ids  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
cfg = PipelineConfig()
BASE_MODEL = cfg.student_model
TRAIN_FILE = os.path.join(cfg.splits_dir, "train.jsonl")
VAL_FILE   = os.path.join(cfg.splits_dir, "val.jsonl")
OUTPUT_DIR_SFT = cfg.output_dir_sft
OUTPUT_DIR_DTW = cfg.output_dir_dtw

LORA_R = cfg.lora_r
LORA_ALPHA = cfg.lora_alpha
LORA_DROPOUT = cfg.lora_dropout
LORA_TARGETS = cfg.lora_targets

EPOCHS = cfg.epochs
PER_DEVICE_BATCH = cfg.per_device_batch
GRAD_ACCUM = cfg.grad_accum
LR = cfg.learning_rate
WARMUP_RATIO = cfg.warmup_ratio
SAVE_STEPS = cfg.save_steps
LOGGING_STEPS = cfg.logging_steps
SEED = cfg.seed

DTW_TAU = cfg.dtw_tau
DTW_SOURCE = cfg.dtw_source


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def to_messages(row):
    out = []
    for m in row["messages"]:
        role = "assistant" if m["role"] == "model" else m["role"]
        out.append({"role": role, "content": m["content"]})
    return out


def _encode_chat(msgs, tokenizer, add_generation_prompt):
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def render_and_mask(row, tokenizer, max_len):
    msgs = to_messages(row)
    full_ids = _encode_chat(msgs, tokenizer, add_generation_prompt=False)
    prompt_only = _encode_chat(msgs[:-1], tokenizer, add_generation_prompt=True)
    boundary = len(prompt_only)
    labels = list(full_ids)
    for i in range(min(boundary, len(labels))):
        labels[i] = -100
    return full_ids[:max_len], labels[:max_len]


# ----------------------------------------------------------------------------
# Phase 1: inspect
# ----------------------------------------------------------------------------
def inspect_lengths(tokenizer, train_file):
    rows = load_jsonl(train_file)
    lengths, resp_lengths = [], []
    for r in rows:
        msgs = to_messages(r)
        full = _encode_chat(msgs, tokenizer, add_generation_prompt=False)
        prompt = _encode_chat(msgs[:-1], tokenizer, add_generation_prompt=True)
        lengths.append(len(full))
        resp_lengths.append(len(full) - len(prompt))
    a = np.array(lengths); r = np.array(resp_lengths)
    print(f"\nToken-length distribution over {len(a)} training rows:")
    for p in (50, 75, 90, 95, 99, 100):
        print(f"  p{p:<3}: {int(np.percentile(a, p)):>5}")
    print(f"  mean: {a.mean():.0f}   max: {a.max()}")
    print("\nResponse-only token length:")
    for p in (50, 90, 99, 100):
        print(f"  p{p:<3}: {int(np.percentile(r, p)):>5}")
    print("\nMAX_LEN candidates (rows truncated at each):")
    for cap in (768, 1024, 1280, 1536, 2048):
        n_trunc = int((a > cap).sum())
        print(f"  max_len={cap:<5}: truncates {n_trunc:>4} / {len(a)} "
              f"({100*n_trunc/len(a):.1f}%)")


# ----------------------------------------------------------------------------
# DTW loss (Ploutos Eq. 1 & 2)
# ----------------------------------------------------------------------------
def _verbalizer_ids(label, tokenizer):
    return get_verbalizer_token_ids(label, tokenizer)


def _compute_type_embedding(model, verbalizer_ids, source):
    import torch
    ids = torch.tensor(verbalizer_ids, device=model.device).unsqueeze(0)
    if source == "embedding":
        return model.get_input_embeddings()(ids).mean(dim=1).squeeze(0)
    elif source == "hidden":
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        return out.hidden_states[-1].mean(dim=1).squeeze(0)
    raise ValueError(f"source must be 'hidden' or 'embedding', got {source!r}")


def _dtw_token_weights(hidden_states, type_emb, response_mask, tau, debug=False):
    """alpha_i (Eq. 2) over the response tokens, done in FP32 since in bf16 the
    softmax goes ~one-hot and the loss blows up. Returns a (T,) tensor that sums
    to 1 over response tokens and is 0 elsewhere."""
    import torch
    import torch.nn.functional as F
    h = F.normalize(hidden_states.float(), dim=-1)
    t = F.normalize(type_emb.float(), dim=-1).unsqueeze(0)
    cos = (h * t).sum(dim=-1)
    logits = cos / tau
    neg_inf = torch.finfo(logits.dtype).min
    logits = torch.where(response_mask, logits, torch.full_like(logits, neg_inf))
    alpha = torch.softmax(logits, dim=0)
    alpha = alpha * response_mask.to(alpha.dtype)
    if debug:
        nz = alpha[response_mask]
        eff = torch.exp(-(nz * (nz + 1e-12).log()).sum()).item()
        cr = cos[response_mask]
        print(f"[DTW] resp_tokens={int(response_mask.sum().item())} "
              f"alpha_max={nz.max().item():.3f} eff_tokens={eff:.1f} "
              f"cos[min/max]={cr.min().item():+.3f}/{cr.max().item():+.3f}  "
              f"(want eff_tokens in the dozens; ~1 = collapsed -> raise tau)",
              flush=True)
    return alpha


def make_dtw_trainer_class():
    from transformers import Trainer
    import torch
    import torch.nn.functional as F

    class DTWTrainer(Trainer):
        def __init__(self, *args, up_ids=None, down_ids=None,
                     dtw_tau=0.5, dtw_source="embedding", debug=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.up_ids = up_ids
            self.down_ids = down_ids
            self.dtw_tau = dtw_tau
            self.dtw_source = dtw_source
            self.debug = debug
            self._dbg_count = 0

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            labels = inputs["labels"]
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
            )
            logits = outputs.logits
            hidden = outputs.hidden_states[-1]
            B, T, V = logits.shape

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_hidden = hidden[:, :-1, :].contiguous()

            ce = F.cross_entropy(
                shift_logits.view(-1, V), shift_labels.view(-1),
                ignore_index=-100, reduction="none").view(B, T - 1)
            response_mask = shift_labels != -100

            t_up = _compute_type_embedding(model, self.up_ids, self.dtw_source)
            t_dn = _compute_type_embedding(model, self.down_ids, self.dtw_source)

            dbg = self.debug and self._dbg_count < 2   # only the first 2 microbatches
            dtw_label = inputs.get("dtw_label")

            # weighted CE sum for this microbatch. alpha sums to 1 per example
            # (a weighted mean), so scale it by k (no. of response tokens) to
            # make it sum to k - then (alpha_k * ce).sum() is on the same scale
            # as the summed token CE the normal Trainer uses
            weighted_ce_sum = shift_logits.new_zeros(())
            total_resp_tokens = 0
            for b in range(B):
                rmask = response_mask[b]
                k = int(rmask.sum().item())
                if k == 0:
                    continue
                is_up = (dtw_label is not None and int(dtw_label[b]) == 1)
                t_yd = t_up if is_up else t_dn
                alpha = _dtw_token_weights(shift_hidden[b], t_yd, rmask,
                                           self.dtw_tau, debug=(dbg and b == 0))
                alpha_k = alpha * k                              # sums to k
                weighted_ce_sum = weighted_ce_sum + (alpha_k.to(ce.dtype) * ce[b]).sum()
                total_resp_tokens += k

            if total_resp_tokens == 0:
                loss = (shift_logits.sum() * 0.0)   # degenerate batch, keep grad
                return (loss, outputs) if return_outputs else loss

            # scaling: transformers passes num_items_in_batch (response tokens
            # across the whole grad accumulation window) but doesn't normalise an
            # overridden loss itself. A plain mean would inflate loss/grads by
            # ~grad_accum and a plain sum by the token count, so divide by
            # num_items_in_batch - summed over the window that gives the proper
            # per-token mean (~1.5)
            if num_items_in_batch is not None:
                loss = weighted_ce_sum / num_items_in_batch
            else:
                loss = weighted_ce_sum / total_resp_tokens

            self._dbg_count += 1
            if dbg:
                per_token_mean = float((weighted_ce_sum / total_resp_tokens).detach())
                print(f"[DTWfix] B={B} resp_tokens={total_resp_tokens} "
                      f"num_items_in_batch={num_items_in_batch} "
                      f"returned_loss={float(loss.detach()):.4f} "
                      f"per_token_mean={per_token_mean:.3f} "
                      f"ce_resp_mean={float(ce[response_mask].detach().mean()):.3f}",
                      flush=True)

            return (loss, outputs) if return_outputs else loss

    return DTWTrainer


def make_stop_callback(max_steps):
    """Stop after `max_steps` optimiser steps but leave the LR schedule alone.
    Setting TrainingArguments.max_steps would squash warmup/cosine into N steps,
    this way a 10 step smoke test logs the same loss/grad_norm as a full run."""
    from transformers import TrainerCallback

    class StopAtStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= max_steps:
                control.should_training_stop = True
            return control

    return StopAtStep()


# ----------------------------------------------------------------------------
# DTW self-test (tiny random model on CPU, no download or HF token needed)
# ----------------------------------------------------------------------------
def dtw_selftest():
    import tempfile
    import warnings
    import torch
    import torch.nn.functional as F
    from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64))
    up_ids, down_ids = [5, 6, 7, 8], [9, 10, 11, 12]

    # 1) alpha is fp32 even from bf16 inputs, sums to 1 over response, 0 elsewhere
    h = torch.randn(20, 64).to(torch.bfloat16)
    t = torch.randn(64).to(torch.bfloat16)
    mask = torch.zeros(20, dtype=torch.bool)
    mask[8:18] = True
    alpha = _dtw_token_weights(h, t, mask, tau=0.5)
    assert alpha.dtype == torch.float32, alpha.dtype
    assert torch.allclose(alpha[mask].sum(), torch.tensor(1.0), atol=1e-5)
    assert alpha[~mask].abs().max() == 0
    print(f"[selftest] alpha ok: fp32, sum={alpha[mask].sum().item():.6f}, "
          f"0 outside response, max={alpha.max().item():.3f}", flush=True)

    # batch of 2, prompt masked, row 1 right-padded, row 0 UP/ row 1 DOWN
    B, T = 2, 24
    input_ids = torch.randint(13, 128, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)
    labels = input_ids.clone()
    labels[0, :10] = -100
    labels[1, :6] = -100
    labels[1, 20:] = -100
    attn[1, 20:] = 0
    inputs = {"input_ids": input_ids, "attention_mask": attn, "labels": labels,
              "dtw_label": torch.tensor([1, 0])}
    n_resp = int((labels[:, 1:] != -100).sum())
    num_items = 4 * n_resp          # pretend this is 1 of 4 microbatches in the window

    DTWTrainer = make_dtw_trainer_class()
    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(output_dir=tmp, use_cpu=True, report_to="none",
                                 remove_unused_columns=False)
        tr = DTWTrainer(model=model, args=args, up_ids=up_ids, down_ids=down_ids,
                        dtw_tau=0.5, dtw_source="embedding", debug=True)

        # 2) loss matches Eq. 1/2 worked out separately, / num_items_in_batch
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loss = tr.compute_loss(model, inputs, num_items_in_batch=num_items)
        grad_warn = [w for w in caught if "requires_grad" in str(w.message)]
        assert not grad_warn, f"requires_grad warning still raised: {grad_warn[0].message}"

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True)
            sl, sh, slab = out.logits[:, :-1], out.hidden_states[-1][:, :-1], labels[:, 1:]
            emb = model.get_input_embeddings()
            ref = 0.0
            for b in range(B):
                m = slab[b] != -100
                ids = up_ids if b == 0 else down_ids
                ty = emb(torch.tensor(ids)).mean(0)
                cos = F.cosine_similarity(sh[b][m].float(), ty.float().unsqueeze(0), dim=-1)
                a = torch.softmax(cos / 0.5, dim=0) * int(m.sum())
                ce = F.cross_entropy(sl[b][m].float(), slab[b][m], reduction="none")
                ref += (a * ce).sum().item()
            ref /= num_items
        assert abs(loss.item() - ref) < 1e-4, (loss.item(), ref)
        print(f"[selftest] DTW loss ok: {loss.item():.6f} == ref {ref:.6f} "
              f"(sum / num_items_in_batch={num_items})", flush=True)

        # 3) gradients flow
        loss.backward()
        g = model.get_input_embeddings().weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        model.zero_grad()
        print("[selftest] backward ok: finite, non-zero grads", flush=True)

        # 4) huge tau -> uniform alpha -> should equal plain summed CE / num_items
        tr.dtw_tau = 1e9
        with torch.no_grad():
            loss_u = tr.compute_loss(model, inputs, num_items_in_batch=num_items)
            ce_sum = F.cross_entropy(sl.reshape(-1, sl.shape[-1]).float(),
                                     slab.reshape(-1), ignore_index=-100,
                                     reduction="sum").item()
        assert abs(loss_u.item() - ce_sum / num_items) < 1e-4, (loss_u.item(), ce_sum / num_items)
        print(f"[selftest] uniform-alpha limit ok: {loss_u.item():.6f} == plain CE "
              f"{ce_sum / num_items:.6f}", flush=True)
        tr.dtw_tau = 0.5

        # 5) num_items_in_batch=None falls back to this microbatch's token count
        with torch.no_grad():
            loss_n = tr.compute_loss(model, inputs, num_items_in_batch=None)
        assert abs(loss_n.item() - ref * num_items / n_resp) < 1e-4
        print("[selftest] num_items_in_batch=None fallback ok", flush=True)

        # 6) no response tokens -> loss of 0 that still has grad
        deg = dict(inputs, labels=torch.full_like(labels, -100))
        loss_d = tr.compute_loss(model, deg, num_items_in_batch=num_items)
        assert loss_d.item() == 0.0 and loss_d.requires_grad
        print("[selftest] degenerate batch ok", flush=True)

        # 7) hidden-source type embedding runs and has the right shape
        th = _compute_type_embedding(model, up_ids, "hidden")
        assert th.shape == (64,)
        print("[selftest] hidden-source type embedding ok", flush=True)

    print("[selftest] ALL OK", flush=True)


# ----------------------------------------------------------------------------
# Phase 2: train
# ----------------------------------------------------------------------------
def train(max_len, loss_type, output_dir, dtw_tau, dtw_source,
          train_file=TRAIN_FILE, val_file=VAL_FILE, debug=False, max_steps=None):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments,
                              DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    dtype = pick_dtype()
    print(describe_device(dtype), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(train_file)
    val_rows = load_jsonl(val_file)

    def build(rs):
        ids, labs, dtw = [], [], []
        n_trunc = 0
        for r in rs:
            i, l = render_and_mask(r, tokenizer, max_len)
            if len(i) == max_len:
                n_trunc += 1
            ids.append(i); labs.append(l)
            dtw.append(1 if r["metadata"].get("ground_truth") == "UP" else 0)
        return Dataset.from_dict(
            {"input_ids": ids, "labels": labs, "dtw_label": dtw}), n_trunc

    train_ds, nt = build(rows)
    val_ds, _ = build(val_rows)
    print(f"train rows: {len(train_ds)} ({nt} hit max_len={max_len})", flush=True)

    sample = train_ds[0]
    n_unmasked = sum(1 for x in sample["labels"] if x != -100)
    print(f"sample row: {len(sample['input_ids'])} tokens, "
          f"{n_unmasked} response tokens scored (rest masked)", flush=True)
    if n_unmasked == 0:
        raise SystemExit("No response tokens unmasked - masking boundary wrong.")
    print("first scored tokens decode to: "
          f"{tokenizer.decode([x for x in sample['labels'] if x != -100][:40])!r}\n",
          flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=dtype)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, padding="longest")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    if loss_type == "dtw":
        up_ids = _verbalizer_ids("UP", tokenizer)
        down_ids = _verbalizer_ids("DOWN", tokenizer)
        print(f"DTW loss: tau={dtw_tau}, source={dtw_source}", flush=True)
        print(f"  verbalizer ids - UP: {up_ids}  DOWN: {down_ids}", flush=True)
        assert up_ids and down_ids and up_ids != down_ids, \
            "verbalizer ids look wrong (empty or identical)"
        DTWTrainer = make_dtw_trainer_class()
        trainer = DTWTrainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=collator,
            up_ids=up_ids, down_ids=down_ids,
            dtw_tau=dtw_tau, dtw_source=dtw_source, debug=debug)
    else:
        trainer = Trainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=collator)

    if max_steps is not None:
        trainer.add_callback(make_stop_callback(max_steps))
        print(f"--max-steps {max_steps}: stopping early, LR schedule unchanged",
              flush=True)

    resume = None
    if os.path.isdir(output_dir):
        cks = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if cks:
            resume = True
            print(f"Resuming from checkpoint in {output_dir}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"\nSaved adapter to {os.path.join(output_dir, 'final')}", flush=True)

    print("\n=== sample generations (val, temp=0) ===", flush=True)
    model.config.use_cache = True
    model.eval()
    for r in val_rows[:2]:
        msgs = to_messages(r)[:-1]
        ids = _encode_chat(msgs, tokenizer, add_generation_prompt=True)
        prompt_ids = torch.tensor([ids], device=model.device)
        attn = torch.ones_like(prompt_ids)
        with torch.no_grad():
            out = model.generate(prompt_ids, attention_mask=attn,
                                 max_new_tokens=400, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(out[0][prompt_ids.shape[1]:],
                               skip_special_tokens=True)
        md = r["metadata"]
        print(f"\n--- {md['ticker']} {md['date']} | truth={md['ground_truth']} ---")
        print(gen[:600], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--dtw-selftest", action="store_true")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--loss", choices=["sft", "dtw"], default="sft")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--train-file", default=TRAIN_FILE)
    ap.add_argument("--val-file", default=VAL_FILE)
    ap.add_argument("--dtw-tau", type=float, default=DTW_TAU)
    ap.add_argument("--dtw-source", choices=["hidden", "embedding"],
                    default=DTW_SOURCE)
    ap.add_argument("--debug", action="store_true",
                    help="print [DTW]/[DTWfix] diagnostics for the first 2 "
                         "microbatches (DTW loss only).")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after N optimizer steps (smoke test). The LR "
                         "schedule still spans the full epochs.")
    args = ap.parse_args()

    if args.dtw_selftest:
        dtw_selftest()
        return

    if args.inspect:
        from transformers import AutoTokenizer
        inspect_lengths(AutoTokenizer.from_pretrained(BASE_MODEL), args.train_file)
        return
    if args.max_len is None:
        raise SystemExit("Run --inspect first, then pass --max-len <N>.")

    output_dir = args.output_dir or (
        OUTPUT_DIR_DTW if args.loss == "dtw" else OUTPUT_DIR_SFT)
    print(f"loss={args.loss}  output_dir={output_dir}  max_len={args.max_len}",
          flush=True)
    train(args.max_len, args.loss, output_dir, args.dtw_tau, args.dtw_source,
          train_file=args.train_file, val_file=args.val_file,
          debug=args.debug, max_steps=args.max_steps)


if __name__ == "__main__":
    main()

# Ploutos: multi-expert LLM stock prediction

This is my reimplementation of the Ploutos pipeline for stock movement prediction.
Three experts look at each stock/ date and give a BUY or SELL call:

- **price/macro** (pm): deterministic, computed straight from price features
- **sentiment** (sn): Gemini scores each recent headline between -1 and +1, the average decides the call
- **fundamental** (fn): Gemini looks at the latest fundamentals from the financial statements

A "rearview-mirror" teacher (Gemini) then writes a rationale for each row while
knowing the actual outcome, and a student model (Llama 3.1 8B, QLoRA) is fine-tuned
on those rationales using either plain SFT or the DTW loss from the paper. At test
time the student only sees the expert inputs, not the rationale or the answer.

Each expert lives in its own module under `experts/` and registers itself, so the
generator, teacher and eval scripts just loop over whatever is in
`config.active_experts`. Adding or dropping an expert is one new file plus a config
change (more on this [below](#expert-architecture)).

## Pipeline

| Stage | Script | What it does |
|-------|--------|--------------|
| 1 | `backtest/generate_labels.py` | forward-return labels over `horizon_days` |
| 2 | `experts/build_fundamental_features.py`, `experts/build_sentiment_features.py`, `experts/build_master_training_dataset.py` | feature building |
| 3 | `experts/expert_signal_generator_batch.py` | expert signals via the Gemini Batch API |
| 4 | `teacher_model_batch.py` | teacher rationales, writes the SFT jsonl |
| 5 | `make_temporal_split.py` | train/ val/ test split by date (not random) |
| 6 | `train/sft_train.py` | QLoRA fine-tune, SFT or DTW loss |
| 7 | `train/eval_student.py` | out-of-sample eval (MCC, broken down by how many experts were right) |

## Running it

Everything is run from the repo root. The scripts load `config.PipelineConfig`
automatically, `--config` is only needed to override the defaults.

```bash
# data downloads (once, before stage 1)
python data_download_scripts/download_spy_data.py
python data_download_scripts/download_news.py
python data_download_scripts/download_fin_statements_alphavantage.py

# 1. labels
python backtest/generate_labels.py

# 2. features
python experts/build_fundamental_features.py
python experts/build_sentiment_features.py
python experts/build_master_training_dataset.py

# 3. expert signals (batch api: create, poll, merge)
python experts/expert_signal_generator_batch.py --mode create --input data/master_training_data.csv
python experts/expert_signal_generator_batch.py --mode poll
python experts/expert_signal_generator_batch.py --mode merge  --input data/master_training_data.csv

# 4. teacher
python teacher_model_batch.py --mode create
python teacher_model_batch.py --mode poll
python teacher_model_batch.py --mode merge

# 5. split (dry run first, then --write)
python make_temporal_split.py data/sft_training_data.jsonl --outdir data/splits
python make_temporal_split.py data/sft_training_data.jsonl --outdir data/splits --write

# 6. training
python train/sft_train.py --inspect                 # check token lengths first
python train/sft_train.py --max-len 1024 --loss sft
python train/sft_train.py --max-len 1024 --loss dtw --dtw-source embedding
python train/sft_train.py --dtw-selftest            # quick check of the DTW maths on a tiny model, cpu only

# 7. eval
python train/eval_student.py --adapter results/checkpoint-100 \
    --test data/splits/test.jsonl --stratified 300 --seed 42

# score each expert on its own (no student involved)
python train/eval_experts.py                 # accuracy + MCC per expert
python train/eval_experts.py --ablation      # plus leave-one-out consensus
python train/eval_experts.py --experts pm,fn # any subset
```

## Running on the cluster

I train and evaluate on an HPC cluster with Slurm. The repo lives in
`/scratch/loz456/ploutos`, the venv in `/scratch/loz456/ploutos-venv` and the HF
cache in `/scratch/loz456/hf_cache`. The baseline adapters in `results/` were
trained on a single 16 GB T4. On an A100 the default setup (4-bit QLoRA, bf16
compute) stays the same. The compute dtype is picked in `core/precision.py` and
printed at startup: bf16 if the GPU supports it (A100), fp16 otherwise (e.g. P100,
which I only use for eval). The DTW token weights are always computed in fp32.

Setup (once):

```bash
cd /scratch/loz456
git clone <repo-url> ploutos && cd ploutos
python3 -m venv /scratch/loz456/ploutos-venv
source /scratch/loz456/ploutos-venv/bin/activate
# install torch first so it matches the node's CUDA (cu126 still supports P100/sm_60)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cp .env.example .env && chmod 600 .env   # add keys incl. HF_TOKEN (Llama is gated)
```

`data/splits/` is in git so a `git pull` brings the train/val/test files across.
The adapter weights (`*.safetensors`) are too big for git, so copy them over once,
e.g. `scp -r results/checkpoint-100 <cluster>:/scratch/loz456/ploutos/out/sft_baseline/`.

Submitting jobs:

```bash
sbatch slurm/train.sbatch sft        # or: dtw [--dtw-source embedding ...]
sbatch slurm/eval.sbatch out/sft_baseline/checkpoint-100
```

Logs go to `logs/<job-name>-<jobid>.out`. Training requeues itself 5 minutes before
the time limit and picks up from the latest `checkpoint-*`. Eval writes predictions
row by row to `<adapter>/eval_test_strat300_seed42.jsonl`, so running the same
command again continues where it stopped.

To syntax check the sbatch scripts use `for f in slurm/*.sbatch; do bash -n "$f"; done`
(`bash -n slurm/*.sbatch` only checks the first file).

One thing to watch out for: if a job gets killed while it is writing a checkpoint,
resume will try to load the half-written folder and crash. Delete the newest
`checkpoint-*` and resubmit.

### Checking a new machine

1. Eval of `checkpoint-100` should roughly reproduce the baseline
   (`reference_results/BASELINE.md`, section 3): overall MCC around -0.026 (anything
   within about +/-0.05 is fine), ec=0 clearly negative (~ -0.35), ec=3 positive
   (~ +0.30), and the student not beating the best simple baseline. The startup
   line should say `bfloat16` on an A100.
2. DTW smoke test: `sbatch slurm/train.sbatch dtw --debug --max-steps 10 --output-dir out/dtw_smoke`.
   The first `[DTWfix]` line should show `returned_loss` around 0.1, and step 10
   should log loss ~1.30 and grad_norm ~0.88 (my original run had 1.3026/ 0.879).
   `--max-steps` stops early without shrinking the LR schedule, so step 10 is
   comparable with a full run. Use a fresh output dir, otherwise it resumes from
   the old checkpoint.
3. On a P100 the startup line should say fp16. Results can move a little compared
   to bf16, so use the A100 for check 1.

## Expert architecture

Each expert is a module in `experts/` that subclasses `DeterministicExpert` or
`LLMExpert` (both in `experts/base.py`) and calls `register()` at the bottom. The
generator, teacher, `compute_experts_correct` and `train/eval_experts.py` all get
their experts from `get_active(cfg)` in `experts/registry.py`, so nothing is
hardcoded to pm/sn/fn.

What an expert defines:

| Member | Meaning |
|--------|---------|
| `id` / `name` | registry key (`"pm"`)/ readable name |
| `signal_column`, `input_column`, `view_column` | the master CSV columns it owns |
| `source_tag`, `prompt_title`, `include_basis` | how its block shows up in the teacher prompt |
| `is_llm`, `persona` | whether it needs an LLM call, and the system prompt if so |
| `can_abstain`, `data_scope` | whether it can say NEUTRAL; `single_ticker`, `cross_sectional` or `external` |
| `build_features(panel, config)` | its feature step, returns columns keyed on `(Ticker, Date)` |
| `make_input(row)` | the text block it adds to the prompt |
| `signal(row)` | its call as UP/ DOWN/ NEUTRAL/ None, used for eval and `experts_correct` |

- `DeterministicExpert` implements `compute(row) -> (label, view)` and `signal()`
  recomputes from the features, so it can be scored without any LLM or GPU.
- `LLMExpert` provides `persona`, `schema` and `parse_response(result)`, and
  `signal()` reads the cached signal column.

The current experts store BUY/SELL, `signal()` maps that to UP/DOWN using each
expert's `vocabulary`. A new expert can just use UP/DOWN directly.

**Data scope.** `build_features` gets the whole panel, so anything that needs other
tickers or outside data is handled there. By the time `make_input`/`signal` see a
row, everything is already a column.

- single ticker (pm/sn/fn): group by `Ticker`
- cross-sectional: group by `Date` and rank against peers, see
  [`experts/examples/sector_relative_momentum.py`](experts/examples/sector_relative_momentum.py)
  (it runs as a demo on its own)
- external data: load your own series (rates, oil, FX etc.) and map it onto
  `(Ticker, Date)` e.g. by sector or beta

**Adding an expert:**

1. Create `experts/my_expert.py` with a subclass and `register(MyExpert())` at the bottom.
2. Import it in `experts/__init__.py`, or import it wherever you use it.
3. Add its `id` to `config.active_experts`.
4. Score it on its own first with `python train/eval_experts.py`.

To drop one, just remove its `id` from `active_experts`.

**Scoring the experts.** `train/eval_experts.py` scores each expert's `signal(row)`
against the test set (it joins the keys in `data/splits/test.jsonl` back to
`master_training_data.csv` for the features and ground truth) and reports accuracy
and MCC, overall and by `experts_correct` bucket. The three current experts all sit
around chance (pm 48.7%, sn 46.9%, fn 51.3%), which matches what I expected. With
`--ablation` it also shows consensus accuracy/ MCC with each expert left out, so you
can see what each one adds without retraining anything.

## Configuration

All parameters are in `config.py` (`PipelineConfig`). Either change the defaults
there or pass a YAML file:

```bash
python make_temporal_split.py data/sft_training_data.jsonl --outdir data/splits \
    --config experiments/horizon30.yaml
```

```yaml
# experiments/horizon30.yaml
horizon_days: 30
news_lookback_days: 14
val_start: "2024-01-01"
test_start: "2024-07-01"
```

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `horizon_days` | 5 | forward-return window for the labels, also the "next N trading days" in the sentiment persona |
| `news_lookback_days` | 7 | calendar days of news given to the sentiment expert (separate from `horizon_days`) |
| `flat_lower_pct` | 35.0 | returns below this percentile are labelled DOWN |
| `flat_upper_pct` | 65.0 | returns above this percentile are labelled UP |
| `val_start` | 2024-01-01 | train/ val boundary |
| `test_start` | 2024-07-01 | val/ test boundary |
| `active_experts` | ["pm","sn","fn"] | which experts are used everywhere |
| `student_model` | meta-llama/Llama-3.1-8B-Instruct | base model for the student |
| `expert_model` | gemini-2.5-flash-lite | model for the LLM experts |
| `teacher_model` | gemini-3-flash-preview | model for the teacher rationales |
| `lora_r` / `lora_alpha` | 16 / 32 | LoRA rank and scaling |
| `dtw_tau` | 0.5 | DTW softmax temperature (Eq. 2 in the paper) |
| `dtw_source` | "embedding" | "embedding" (better) or "hidden" for the type embedding |
| `earnings_lag_days` | 45 | filing date is approximated as fiscalDateEnding + N days |
| `sentiment_highly_threshold` | 0.35 | FinBERT score above this counts as Highly Bullish/ Bearish |

API keys go in a `.env` file in the repo root (see `.env.example`), loaded by
`core/secrets.py`. Hugging Face picks up `HF_TOKEN` from the environment by itself.

## Tests

```bash
pytest tests/ -v
python train/sft_train.py --dtw-selftest   # DTW loss against hand-computed Eq. 1/2, cpu
```

The tests cover the `extract_direction` cases from `reference_results/BASELINE.md`,
the DTW word lists, the split counts (1104/ 277/ 819) and date boundaries, label
balance, and that the expert modules reproduce the stored signals, views and prompt
inputs exactly (`test_price_macro_byte_identical.py`, `test_llm_experts_offline.py`).
`test_eval_experts.py` checks every expert stays near chance on its own.

Tests that need `data/master_training_data.csv` or `data/sft_training_data.jsonl`
are skipped if those files aren't there.

## Things not to change

A few details that the results depend on:

1. **Loss masking** (`core/io_utils.py:encode_chat`): render the chat to a string
   with `apply_chat_template(..., tokenize=False)` first, then tokenize. Passing the
   message list straight to the tokenizer returns a `tokenizers.Encoding` and the
   response offset ends up wrong.
2. **fp32 in the DTW weights** (`train/sft_train.py:_dtw_token_weights`): the
   `.float()` casts on the hidden states and type embedding are needed. In bf16 the
   softmax over ~300 response tokens collapses to almost one-hot and the loss blows
   up (~20 instead of ~1.6).
3. **DTW anchor words** (`core/verbalizer.py`): `DTW_UP_WORDS` and `DTW_DOWN_WORDS`
   are 4 words each, as in Appendix C of the paper. Adding more words pulls the UP
   and DOWN anchors closer together in embedding space.
4. **Temporal split**: always split by date, never randomly, and don't tune on the
   test set.

## Baseline

`reference_results/BASELINE.md` has the reference numbers. In short:

- split: train 1104/ val 277/ test 819
- overall MCC -0.026 on 300 stratified test rows (checkpoint-100, epoch 1.45)
- ec=0 clearly negative, ec=3 clearly positive, i.e. the student mostly follows the
  experts when they agree and doesn't learn much beyond that

## Known issues

- The news block header in the sentiment prompt says `past 5 days` but the window
  is actually `news_lookback_days` (7 calendar days by default). Fixing it changes
  every prompt, so it would mean regenerating the teacher data and a new baseline.
- The sentiment persona contains the literal text `{_HORIZON_DAYS}` because it was
  never `.format()`-ed, so Gemini sees the braces as they are. Same as above, I'll
  only fix this together with a full regeneration.

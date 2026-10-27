# Baseline results

Reference numbers for the pipeline so I can check changes against something
written down instead of from memory. Recorded on 21/06/2026.

The adapter used for sections 3 and 4 is `results/checkpoint-100` (global_step 100,
epoch 1.45). The only other saved checkpoint is checkpoint-50 (epoch 0.72).
Training was set to 207 steps but stopped at step 100.

## 1. Temporal split (should match exactly)

Source: `data/sft_training_data.jsonl` (2200 rows), split as
train < 2024-01-01 <= val < 2024-07-01 <= test

    train : 1104 rows   (2022-01-03 .. 2023-12-29)
    val   :  277 rows   (2024-01-02 .. 2024-06-28)
    test  :  819 rows   (2024-07-01 .. 2025-12-31)

Label balance:

    train: DOWN 574 / UP 530
    val  : DOWN 137 / UP 140
    test : DOWN 386 / UP 433

experts_correct spread (train): {0: 216, 1: 379, 2: 344, 3: 165}

These are checked in `tests/test_split.py`.

## 2. extract_direction (should match exactly)

    "D.R. Horton is expected to rise"                         -> UP
    "A.O. Smith is expected to decline sharply"               -> DOWN
    "The stock is unlikely to decline and should rise"        -> UP
    "There is little downside from here"                      -> UP
    "We see limited upside in the near term"                  -> DOWN
    "The stock will rise, though some decline is possible"    -> UP
    "The price is expected to see a downward movement"        -> DOWN
    (long opener with no direction)                           -> None

These are checked in `tests/test_verbalizer.py`.

## 3. Plain SFT student, out-of-sample eval

Stratified 300 rows of `data/splits/test.jsonl`, seed 42. 256 rows parsed.

    STUDENT OVERALL          acc 47.7%   MCC -0.026

    by experts_correct bucket:
      ec=0   n=63   acc 34.9%   MCC -0.350
      ec=1   n=74   acc 51.4%   MCC +0.030
      ec=2   n=74   acc 48.6%   MCC -0.030
      ec=3   n=45   acc 57.8%   MCC +0.299

    BASELINES (same rows):
      always-UP                acc 53.5%   MCC  0.000
      always-DOWN              acc 46.5%   MCC  0.000
      follow-expert-majority   acc 49.4%   MCC -0.024

    Student agrees with expert-majority on 54.8% of rows.

Generation at temperature 0 can still move a bit between library versions and
GPUs, so I don't expect identical numbers on a re-run. What should hold:

- overall MCC close to zero (roughly -0.05 to +0.05)
- ec=0 clearly negative and ec=3 clearly positive (the student follows the experts
  when they agree)
- the student doesn't beat the best simple baseline

If a re-run suddenly shows something like overall MCC +0.2, something has changed
and needs looking into.

## 4. DTW run health check

A normal DTW run (`--loss dtw --dtw-source embedding`, fp32 weights) should show on
the first `[DTW]` line:

- eff_tokens in the dozens (around 1 means the weights collapsed)
- a starting loss somewhere around 1.5 to 2.5, not ~20

The ~20.8 loss I saw early on came from computing the weights in bf16, which is why
the fp32 cast is there.

"""
Tests for the train/ val/ test date split, counts are from
reference_results/BASELINE.md section 1.
Run from the repo root with: pytest tests/test_split.py -v
Needs data/sft_training_data.jsonl (2200 rows), skipped if it's missing.
"""
import json
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PipelineConfig

SFT_JSONL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sft_training_data.jsonl",
)


def _load_and_split():
    cfg = PipelineConfig()
    val_start  = date.fromisoformat(cfg.val_start)
    test_start = date.fromisoformat(cfg.test_start)

    train, val, test = [], [], []
    with open(SFT_JSONL) as f:
        for line in f:
            row = json.loads(line)
            d = date.fromisoformat(row["metadata"]["date"])
            if d < val_start:
                train.append(row)
            elif d < test_start:
                val.append(row)
            else:
                test.append(row)
    return train, val, test


@pytest.fixture(scope="module")
def splits():
    if not os.path.exists(SFT_JSONL):
        pytest.skip("data/sft_training_data.jsonl not present")
    return _load_and_split()


def test_split_counts(splits):
    train, val, test = splits
    assert len(train) == 1104, f"train={len(train)}"
    assert len(val)   ==  277, f"val={len(val)}"
    assert len(test)  ==  819, f"test={len(test)}"


def test_split_date_boundaries(splits):
    cfg = PipelineConfig()
    val_start  = date.fromisoformat(cfg.val_start)
    test_start = date.fromisoformat(cfg.test_start)

    train, val, test = splits
    train_dates = [date.fromisoformat(r["metadata"]["date"]) for r in train]
    val_dates   = [date.fromisoformat(r["metadata"]["date"]) for r in val]
    test_dates  = [date.fromisoformat(r["metadata"]["date"]) for r in test]

    assert max(train_dates) < val_start
    assert min(val_dates)   >= val_start
    assert max(val_dates)   < test_start
    assert min(test_dates)  >= test_start


def test_split_ground_truth_balance(splits):
    train, val, test = splits

    def counts(rows):
        up   = sum(1 for r in rows if r["metadata"].get("ground_truth") == "UP")
        down = sum(1 for r in rows if r["metadata"].get("ground_truth") == "DOWN")
        return up, down

    up, dn = counts(train)
    assert (dn, up) == (574, 530), f"train balance DOWN={dn}/UP={up}"

    up, dn = counts(val)
    assert (dn, up) == (137, 140), f"val balance DOWN={dn}/UP={up}"

    up, dn = counts(test)
    assert (dn, up) == (386, 433), f"test balance DOWN={dn}/UP={up}"


def test_split_experts_correct_spread(splits):
    train, _, _ = splits
    from collections import Counter
    ec = Counter(r["metadata"].get("experts_correct") for r in train)
    assert dict(ec) == {0: 216, 1: 379, 2: 344, 3: 165}

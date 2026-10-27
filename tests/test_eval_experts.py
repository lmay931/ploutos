"""
Sanity check for the per-expert standalone OOS eval.

Each of the three experts on its own should be near chance on the test set
(~49%, mcc ~0). If signal sourcing breaks, an expert ends up at 0%/ 100% or NaN
instead. Runs offline (pm recomputed, sn/fn from cached signals), skips if the
master CSV or test split isn't there.
"""
import os
import sys

try:
    import pytest
except ImportError:
    class _PytestShim:
        class _Skip(Exception):
            pass

        def skip(self, msg):
            raise self._Skip(msg)
    pytest = _PytestShim()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PipelineConfig                      # noqa: E402
from core.metrics import confusion                     # noqa: E402
from experts.registry import get_active                # noqa: E402
from train.eval_experts import _load_test_rows, DIRECTIONS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "data", "master_training_data.csv")
TEST = os.path.join(ROOT, "data", "splits", "test.jsonl")


def _rows_active():
    if not (os.path.exists(MASTER) and os.path.exists(TEST)):
        pytest.skip("master CSV or test split not found")
    rows = _load_test_rows(MASTER, TEST)
    if len(rows) == 0:
        pytest.skip("no test rows after join")
    return rows, get_active(PipelineConfig())


def test_each_expert_is_near_chance_standalone():
    rows, active = _rows_active()
    row_list = [r for _, r in rows.iterrows()]
    summary = {}
    for e in active:
        pairs = [(e.signal(r), r["Target_Label"]) for r in row_list
                 if e.signal(r) in DIRECTIONS]
        assert pairs, f"{e.id} made no directional calls - signal sourcing broke"
        tp, tn, fp, fn = confusion(pairs)
        acc = (tp + tn) / len(pairs)
        summary[e.id] = (len(pairs), acc)
        # a weak but correctly wired expert should sit roughly in 0.40-0.60
        assert 0.40 <= acc <= 0.60, (
            f"{e.id} overall acc {acc:.3f} outside near-chance band - the "
            f"known result is ~49%; sourcing or normalization likely broke.")
    print("\nper-expert standalone (n, acc):", summary)


if __name__ == "__main__":
    test_each_expert_is_near_chance_standalone()
    print("eval_experts sanity check passed.")

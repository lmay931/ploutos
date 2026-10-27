"""
All the pipeline settings in one place.

Defaults:
    from config import PipelineConfig
    cfg = PipelineConfig()

Or override some of them from a YAML file for an experiment:
    cfg = PipelineConfig.from_yaml("experiments/horizon30.yaml")

where the YAML just has field names at the top level, e.g.
    horizon_days: 30
    val_start: "2024-06-01"

and cfg.to_yaml("my_run.yaml") saves the current settings.

Groups:
Horizon          : horizon_days, news_lookback_days
                   (separate settings - horizon_days is the forward return
                   window so it changes the labels, news_lookback_days is how
                   far back the sentiment expert looks)
Label thresholds : flat_lower_pct, flat_upper_pct
Temporal split   : data_start, data_end, val_start, test_start
Expert selection : active_experts  (keys into EXPERT_SIGNAL_COLS in teacher script)
Models           : expert_model, teacher_model, student_model, teacher_temperature
LoRA             : lora_r, lora_alpha, lora_dropout, lora_targets
Training         : epochs, per_device_batch, grad_accum, learning_rate,
                   warmup_ratio, save_steps, logging_steps, seed
DTW              : dtw_tau, dtw_source
Teacher gen      : teacher_sample_size, teacher_seed, drop_direction_mismatches
Eval             : max_new_tokens
Fundamental feat : earnings_lag_days, ttm_window, roe_plausibility_cap
Sentiment feat   : sentiment_highly_threshold, sentiment_cautiously_threshold
Paths            : data_dir, splits_dir, sft_jsonl, master_csv,
                   output_dir_sft, output_dir_dtw
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PipelineConfig:
    # ----------------------------------------------------------------
    # Horizon - label window and the news window mentioned in prompts.
    # The two are set separately, changing one doesn't change the other
    # ----------------------------------------------------------------
    horizon_days: int = 5
    news_lookback_days: int = 7

    # ----------------------------------------------------------------
    # Label thresholds (percentiles of returns for the UP/DOWN/FLAT split)
    # ----------------------------------------------------------------
    flat_lower_pct: float = 35.0   # bottom N% -> DOWN
    flat_upper_pct: float = 65.0   # top (100-N)% -> UP

    # ----------------------------------------------------------------
    # Temporal split boundaries
    # ----------------------------------------------------------------
    data_start: str = "2022-01-01"   # master dataset window start
    data_end: str = "2025-12-31"     # master dataset window end
    val_start: str = "2024-01-01"    # train | val boundary
    test_start: str = "2024-07-01"   # val  | test boundary

    # ----------------------------------------------------------------
    # Expert selection
    # keys into EXPERT_SIGNAL_COLS in teacher_model_batch.py. Taking one out
    # means that expert isn't counted in experts_correct
    # ----------------------------------------------------------------
    active_experts: List[str] = field(
        default_factory=lambda: ["pm", "sn", "fn"]
    )

    # ----------------------------------------------------------------
    # Models and temperatures
    # ----------------------------------------------------------------
    expert_model: str = "gemini-2.5-flash-lite"
    teacher_model: str = "gemini-3-flash-preview"
    student_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    teacher_temperature: float = 1.0

    # ----------------------------------------------------------------
    # LoRA
    # ----------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    epochs: int = 3
    per_device_batch: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    save_steps: int = 50
    logging_steps: int = 10
    seed: int = 42

    # ----------------------------------------------------------------
    # DTW loss (Ploutos Eq. 1 & 2)
    # dtw_source: "embedding" (use this) or "hidden" (collapses on Llama 3 8B)
    # ----------------------------------------------------------------
    dtw_tau: float = 0.5
    dtw_source: str = "embedding"

    # ----------------------------------------------------------------
    # Teacher generation
    # ----------------------------------------------------------------
    teacher_sample_size: int = 5000
    teacher_seed: int = 33
    drop_direction_mismatches: bool = False

    # ----------------------------------------------------------------
    # Eval
    # ----------------------------------------------------------------
    max_new_tokens: int = 400

    # ----------------------------------------------------------------
    # Fundamental feature parameters
    # ----------------------------------------------------------------
    earnings_lag_days: int = 45     # fiscalDateEnding + this ~ Filing_Date
    ttm_window: int = 4             # quarters in the trailing 12 month sums
    roe_plausibility_cap: float = 2.5  # |ROE| above this set to null (distorted by buybacks)

    # ----------------------------------------------------------------
    # Sentiment feature parameters
    # ----------------------------------------------------------------
    sentiment_highly_threshold: float = 0.35    # |score| >= this -> Highly Bull/Bear
    sentiment_cautiously_threshold: float = 0.10  # |score| >= this -> Cautiously Bull/Bear

    # ----------------------------------------------------------------
    # Paths (relative to repo root, or absolute if you prefer)
    # ----------------------------------------------------------------
    data_dir: str = "data"
    splits_dir: str = "data/splits"
    sft_jsonl: str = "data/sft_training_data.jsonl"
    master_csv: str = "data/master_training_data.csv"
    output_dir_sft: str = "out/sft_baseline"
    output_dir_dtw: str = "out/dtw"

    # ----------------------------------------------------------------
    # YAML I/O
    # ----------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load from YAML, keys that aren't field names are ignored (with a warning)."""
        import yaml  # pyyaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = set(data) - set(valid)
        if unknown:
            import warnings
            warnings.warn(f"PipelineConfig.from_yaml: unknown keys ignored: {unknown}")
        return cls(**valid)

    def to_yaml(self, path: str | Path) -> None:
        """Save the current settings to YAML."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False, sort_keys=True)

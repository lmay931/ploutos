# old import path for the API keys and model names, kept so older scripts
# still run. Keys are in core/secrets.py and everything else is in
# config.PipelineConfig - can delete this once nothing imports it.

from core.secrets import (  # noqa: F401
    FINNHUB_API_KEY,
    GEMINI_API_KEY,
    TIINGO_API_KEY,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPHA_VANTAGE_API_KEY,
)

# old model-name constants, just the PipelineConfig defaults
from config import PipelineConfig as _PipelineConfig
_cfg = _PipelineConfig()
GEMINI_BASIC_MODEL_NAME   = _cfg.expert_model
GEMINI_ADVANCED_MODEL_NAME = _cfg.teacher_model
TEACHER_TEMPERATURE       = _cfg.teacher_temperature
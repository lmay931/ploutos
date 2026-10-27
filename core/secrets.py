# API keys, loaded from .env in the repo root (see .env.example). Anything
# already set in the environment (e.g. by slurm/*.sbatch) wins over .env.
# Model names and other settings go in config.PipelineConfig, not here.
#
# HF_TOKEN doesn't need a constant - huggingface_hub reads it from the
# environment, so importing this module is enough for model downloads.
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

FINNHUB_API_KEY       = os.environ.get("FINNHUB_API_KEY")
GEMINI_API_KEY        = os.environ.get("GEMINI_API_KEY")
TIINGO_API_KEY        = os.environ.get("TIINGO_API_KEY")
ALPACA_API_KEY        = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY     = os.environ.get("ALPACA_SECRET_KEY")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

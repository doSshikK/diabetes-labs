"""Synthetic-data generator for laboratory work 5."""

from .config import Lab5Config, load_config
from .generator import generate, run_lab5

__all__ = ["Lab5Config", "generate", "load_config", "run_lab5"]

"""Deep-learning pipeline for laboratory work 3."""

from .config import Lab3Config, load_config
from .pipeline import run_lab3

__all__ = ["Lab3Config", "load_config", "run_lab3"]

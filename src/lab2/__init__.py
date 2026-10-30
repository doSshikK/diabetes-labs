"""Reproducible classical ML pipeline for laboratory work 2."""

from .config import Lab2Config, load_config
from .pipeline import run_lab2

__all__ = ["Lab2Config", "load_config", "run_lab2"]

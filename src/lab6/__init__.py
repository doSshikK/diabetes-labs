"""FastAPI model service for laboratory work 6."""

from .service import ModelService, create_app, create_monitoring_app

__all__ = ["ModelService", "create_app", "create_monitoring_app"]

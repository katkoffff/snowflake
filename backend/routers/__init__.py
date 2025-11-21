# backend/routers/__init__.py
from .init import router as init_router
from .segment import router as segment_router
from .results import router as results_router
from .control import router as control_router
from .analysis import router as analysis_router

__all__ = [
    "init_router",
    "segment_router",
    "results_router",
    "control_router",
    "analysis_router",
]
# backend/schemas/request.py
"""
Pydantic модели для запросов.
"""

from pydantic import BaseModel
from typing import List


class MiniatureData(BaseModel):
    image_path: str
    image_file: str
    display_x: float
    display_y: float
    display_width: float
    display_height: float
    dot_x: float
    dot_y: float
    svg_x: float
    svg_y: float


class PointData(BaseModel):
    x: float
    y: float
    color: str = "blue"


class AxesData(BaseModel):
    x_label: str = "Normalized Perimeter (L/Lc)"
    y_label: str = "Normalized Area (S/Sc)"
    x_range: List[float] = [0, 1]
    y_range: List[float] = [0, 1]


class SaveChartRequest(BaseModel):
    points: List[PointData]
    axes: AxesData
    miniatures: List[MiniatureData]
    viewport_size: dict  # {width: int, height: int}


class SaveToStage2Request(BaseModel):
    folder_name: str
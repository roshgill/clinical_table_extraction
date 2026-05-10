"""Pydantic schemas for router page classification."""

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x_min: float  # left edge, 0–100 percentage
    y_min: float  # top edge, 0–100 percentage
    x_max: float  # right edge, 0–100 percentage
    y_max: float  # bottom edge, 0–100 percentage


class TableClassification(BaseModel):
    label: str          # e.g. "Table 1", "Figure 2A", "top forest plot"
    type: str           # "forest_plot" or "general_table"
    description: str    # one sentence
    instruction: str    # extraction instruction for the sub-agent
    bbox: BoundingBox | None = None  # bounding box in percentage coords


class PageClassification(BaseModel):
    tables: list[TableClassification]

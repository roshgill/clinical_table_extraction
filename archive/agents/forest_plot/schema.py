"""Pydantic schemas for forest plot extraction."""

from pydantic import BaseModel

class ForestPlotExtraction(BaseModel):
    headers: list[str]
    rows: list[list[str]]
    footer: list[str]
    plot_appears_complete: bool

class PageForestPlotResult(BaseModel):
    forest_plots: list[ForestPlotExtraction]

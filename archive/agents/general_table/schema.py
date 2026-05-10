"""Pydantic schemas for general table extraction."""

from pydantic import BaseModel


class TableSegment(BaseModel):
    headers: list[str]
    rows: list[list[str]]
    table_appears_complete: bool


class PageExtractionResult(BaseModel):
    tables: list[TableSegment]

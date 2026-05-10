"""Pydantic schemas for simple table extraction."""

from pydantic import BaseModel


class SimpleTableSegment(BaseModel):
    headers: list[str]
    rows: list[list[str]]
    table_appears_complete: bool


class PageSimpleTableResult(BaseModel):
    tables: list[SimpleTableSegment]

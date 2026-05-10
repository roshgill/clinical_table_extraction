"""Simple table extraction logic."""

import pandas as pd
from PIL import Image

from .prompt import SIMPLE_TABLE_PROMPT
from .schema import PageSimpleTableResult


def extract_simple_tables_from_page(
    client,
    model: str,
    image: Image.Image,
    continuation_headers: list[str] | None = None,
) -> PageSimpleTableResult:
    """Send a page image to Gemini and extract simple tables.

    Args:
        client: google.genai.Client instance.
        model: Model name string.
        image: PIL image of the page.
        continuation_headers: If the last table on the previous page was
            incomplete, pass its headers here so Gemini knows to continue it.
    """
    raise NotImplementedError("Simple table agent not yet implemented")


def stitch_simple_table_results(
    page_results: dict[int, PageSimpleTableResult],
) -> list[pd.DataFrame]:
    """Merge page results into separate DataFrames per logical table.

    Returns a list of DataFrames, one per logical table.
    """
    raise NotImplementedError("Simple table stitching not yet implemented")

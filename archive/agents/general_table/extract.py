"""General table extraction logic."""

import pandas as pd
from PIL import Image

from .prompt import GENERAL_TABLE_PROMPT
from .schema import PageExtractionResult


def extract_tables_from_page(
    client,
    model: str,
    image: Image.Image,
    continuation_headers: list[str] | None = None,
    instruction: str | None = None,
) -> PageExtractionResult:
    """Send a page image to Gemini and extract all tables.

    Args:
        client: google.genai.Client instance.
        model: Model name string.
        image: PIL image of the page.
        continuation_headers: If the last table on the previous page was
            incomplete, pass its headers here so Gemini knows to continue it.
    """
    prompt = GENERAL_TABLE_PROMPT
    if instruction:
        prompt += f"\n\nSPECIFIC INSTRUCTION: {instruction}"
    if continuation_headers:
        prompt += f"""
NOTE: The first table on this page is a CONTINUATION of a table from the previous page.
Its headers are: {continuation_headers}
For that table, return an empty headers list and continue extracting its rows.
Each row for that table must have exactly {len(continuation_headers)} values.
Any additional tables after it on this page should be treated as new tables with their own headers."""

    response = client.models.generate_content(
        model=model,
        contents=[prompt, image],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": PageExtractionResult.model_json_schema(),
        },
    )

    return PageExtractionResult.model_validate_json(response.text)


def stitch_page_results(
    page_results: dict[int, PageExtractionResult],
) -> list[pd.DataFrame]:
    """Merge page results into separate DataFrames per logical table.

    Walks segments in page order. A segment with headers starts a new table.
    A segment without headers is a continuation of the previous table.

    Returns a list of DataFrames, one per logical table.
    """
    all_tables: list[tuple[list[str], list[list[str]]]] = []
    current_headers = None
    current_rows: list[list[str]] = []

    for page_num in sorted(page_results.keys()):
        for segment in page_results[page_num].tables:
            if segment.headers:
                # New table — save previous if it exists
                if current_headers is not None:
                    all_tables.append((current_headers, current_rows))
                current_headers = segment.headers
                current_rows = list(segment.rows)
            else:
                # Continuation of current table
                current_rows.extend(segment.rows)

    # Don't forget the last table
    if current_headers is not None:
        all_tables.append((current_headers, current_rows))

    # Convert to DataFrames
    return [pd.DataFrame(rows, columns=headers) for headers, rows in all_tables]

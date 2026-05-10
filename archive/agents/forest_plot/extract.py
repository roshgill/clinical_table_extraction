"""Forest plot extraction logic."""

import pandas as pd
from PIL import Image

from .prompt import FOREST_PLOT_PROMPT
from .schema import PageForestPlotResult


def extract_forest_plots_from_page(
    client,
    model: str,
    image: Image.Image,
    continuation_context: dict | None = None,
    instruction: str | None = None,
) -> PageForestPlotResult:
    """Send a page image to Gemini and extract forest plot tables.

    Args:
        client: google.genai.Client instance.
        model: Model name string.
        image: PIL image of the page.
        continuation_context: dict with 'title' and 'headers'
            from the previous page's incomplete forest plot, or None.
    """
    prompt = FOREST_PLOT_PROMPT
    if instruction:
        prompt += f"\n\nSPECIFIC INSTRUCTION: {instruction}"
    if continuation_context:
        prompt += f"""\n\nNOTE: The first forest plot on this page is a CONTINUATION from the previous page.
Title: {continuation_context['title']}
Headers: {continuation_context['headers']}

For that forest plot, use the title and headers above (return them as-is).
Each row must have exactly {len(continuation_context['headers'])} values.
Any additional forest plots after it on this page should be treated as new entries."""

    response = client.models.generate_content(
        model=model,
        contents=[prompt, image],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": PageForestPlotResult.model_json_schema(),
        },
    )

    return PageForestPlotResult.model_validate_json(response.text)


def stitch_forest_plot_results(
    page_results: dict[int, PageForestPlotResult],
) -> list[tuple[str, pd.DataFrame]]:
    """Merge page results into DataFrames — one per logical forest plot.

    A forest plot with plot_appears_complete=False on one page is continued
    by the first forest plot on the next page (which shares the same title/headers).

    Returns a list of (title, DataFrame, footer) tuples.
    """
    all_plots: list[tuple[str, list[str], list[list[str]], list[str]]] = []
    current_label = None
    current_headers = None
    current_rows: list[list[str]] = []
    current_footer: list[str] = []
    plot_counter = 0

    for page_num in sorted(page_results.keys()):
        for fp in page_results[page_num].forest_plots:
            if current_label is not None and not fp.headers:
                # Continuation — append rows and update footer
                current_rows.extend(fp.rows)
                current_footer = fp.footer
            else:
                # New forest plot — save previous if exists
                if current_label is not None:
                    all_plots.append((current_label, current_headers, current_rows, current_footer))
                plot_counter += 1
                current_label = f"p{page_num}_plot{plot_counter}"
                current_headers = fp.headers
                current_rows = list(fp.rows)
                current_footer = fp.footer

    # Don't forget the last one
    if current_label is not None:
        all_plots.append((current_label, current_headers, current_rows, current_footer))

    # Convert to DataFrames
    results = []
    for label, headers, rows, footer in all_plots:
        df = pd.DataFrame(rows, columns=headers)
        results.append((label, df, footer))

    return results

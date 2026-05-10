"""PDF-to-image rendering — shared across all agents."""

from pdf2image import convert_from_path
from PIL import Image


def render_pages(pdf_path: str, pages: list[int], dpi: int = 400) -> dict[int, Image.Image]:
    """Render specific PDF pages as PIL images.

    Args:
        pdf_path: Path to the PDF file.
        pages: 1-indexed page numbers to render.
        dpi: Resolution for rendering.

    Returns:
        Dict mapping page number to PIL Image.
    """
    page_images = {}
    for page_num in pages:
        images = convert_from_path(
            pdf_path, first_page=page_num, last_page=page_num, dpi=dpi
        )
        page_images[page_num] = images[0]
    return page_images

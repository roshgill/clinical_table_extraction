"""Router agent — classifies table types found on a page."""

from PIL import Image

from .prompt import ROUTER_PROMPT
from .schema import PageClassification


def classify_page(
    client,
    model: str,
    image: Image.Image,
) -> PageClassification:
    """Classify tables/figures found on a PDF page image.

    For bounding box output, never include titles and labels surrounding the table. Only the table itself.

    Args:
        client: google.genai.Client instance.
        model: Model name string.
        image: PIL image of the page.

    Returns:
        PageClassification with a list of TableClassification entries.
    """
    response = client.models.generate_content(
        model=model,
        contents=[ROUTER_PROMPT, image],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": PageClassification.model_json_schema(),
        },
    )

    return PageClassification.model_validate_json(response.text)

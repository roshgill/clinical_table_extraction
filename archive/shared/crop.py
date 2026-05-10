"""Crop a PIL image using percentage-based bounding box coordinates."""

from PIL import Image


def crop_image(
    image: Image.Image,
    bbox: dict,
    padding_pct: float = 2.0,
) -> Image.Image:
    """Crop image to a bounding box given in percentage coordinates.

    Args:
        image: Source PIL image.
        bbox: Dict with keys x_min, y_min, x_max, y_max (0–100 percentages).
        padding_pct: Extra padding to add around the box, as a percentage of
            the image dimension.  Defaults to 2%.

    Returns:
        Cropped PIL image.
    """
    w, h = image.size
    pad_x = w * padding_pct / 100
    pad_y = h * padding_pct / 100

    left = max(0, w * bbox["x_min"] / 100 - pad_x)
    upper = max(0, h * bbox["y_min"] / 100 - pad_y)
    right = min(w, w * bbox["x_max"] / 100 + pad_x)
    lower = min(h, h * bbox["y_max"] / 100 + pad_y)

    return image.crop((int(left), int(upper), int(right), int(lower)))

import os
from typing import TYPE_CHECKING, Optional, Any
from PIL import Image
if TYPE_CHECKING:
    from dbdie_ml.classes import Boxes, AllSnippetCoords, PlayerId


class SnippetGenerator:
    """Generates player snippets from a full-screen screenshot"""
    def __init__(self, snippet_coords: dict["PlayerId", Any]) -> None:
        self.snippet_coords = snippet_coords

    def _crop_image(
        img: Image,
        boxes: "Boxes",
        offset: Optional[int]
    ) -> None:
        if isinstance(boxes, list):
            using_boxes = boxes
        else:
            if plain.endswith("_4"):
                using_boxes = boxes["killer"]
            else:
                using_boxes = boxes["surv"]

        o = offset if isinstance(offset, int) else 0
        for i, box in enumerate(using_boxes):
            cropped = img.crop(box)
            cropped.save(os.path.join(dst_fd, f"{plain}_{i+o}.jpg"))

    def get_snippets(self, img: Image) -> "AllSnippetCoords":
        img = img.convert("RGB")
        return {
            i: ...
            for i in range(5)
        }

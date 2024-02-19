from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from PIL import Image
    from dbdie_ml.classes import SnippetCoords, SnippetInfo

MOCK_SNIPPETS_COORDS = [
    (67 + i * 117, 217, 67 + (i + 1) * 117, 897)
    for i in range(5)
]


class MockSnippetModel:
    def predict(img: "Image") -> List["SnippetCoords"]:
        return MOCK_SNIPPETS_COORDS


class MockKillerModel:
    def predict(snippet: "SnippetCoords") -> "SnippetInfo":
        return (
            10,  # character
            (10, 11, 12, 13),  # perks
            1,  # item
            (10, 11),  # addons
            2,  # offering
            2,  # status
            10_000  # points
        )


class MockPlayerModel:
    def predict(snippet: "SnippetCoords") -> "SnippetInfo":
        return (
            40,  # character
            (10, 11, 12, 13),  # perks
            1,  # item
            (10, 11),  # addons
            2,  # offering
            2,  # status
            10_000  # points
        )

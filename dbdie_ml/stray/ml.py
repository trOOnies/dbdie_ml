from typing import TYPE_CHECKING, List
from dbdie_ml.schemas import PlayerOut, MatchOut
from backbone.io_requests import get_all_info
from backbone.ml.classes import SnippetCoords, SnippetDetector, SnippetProcessor, EncodedInfo
from backbone.ml.mock import MockSnippetModel, MockKillerModel, MockPlayerModel
if TYPE_CHECKING:
    from PIL import Image

SNIPPET_MODEL = SnippetDetector(model=MockSnippetModel)
KILLER_MODEL = SnippetProcessor(model=MockKillerModel)
SURV_MODEL = SnippetProcessor(model=MockPlayerModel)

# -----


def postprocess_asc(asc: List[SnippetCoords]) -> List[SnippetCoords]:
    return asc


def get_all_snippets_coords(img: "Image") -> List[SnippetCoords]:
    asc = SNIPPET_MODEL.predict(img)
    return postprocess_asc(asc)


def predict_info(snippet: "Image", id: int) -> List[int]:
    if id == 4:
        model = KILLER_MODEL
    else:
        model = SURV_MODEL
    return model.predict(snippet)


def process_snippet(img: "Image", sc: SnippetCoords, id: int) -> EncodedInfo:
    img_snippet = img.crop(sc)
    encoded_info = predict_info(img_snippet, id=id)
    return encoded_info


# -----


async def process_image(img: "Image") -> MatchOut:
    all_snippet_coords = get_all_snippets_coords(img)
    assert len(all_snippet_coords) == 5
    encoded_info = [
        process_snippet(img, sc, id)
        for id, sc in enumerate(all_snippet_coords)
    ]
    all_info = await get_all_info(encoded_info)
    players = [
        PlayerOut(
            id=id,
            character=all_info["characters"][id],
            perks=all_info["perks"][id],
            item=all_info["item"][id],
            addons=all_info["addons"][id],
            offering=all_info["offering"][id],
            status=all_info["status"][id],
            points=all_info["points"][id]
        ) for id in range(len(all_info["characters"]))
    ]
    return MatchOut(players=players)

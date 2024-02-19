import requests
from dbdie_ml.classes import SnippetInfo, PlayerId
from dbdie_ml.schemas import PlayerIn, PlayerOut

URL = "http://127.0.0.1:8000"


def to_player(id: PlayerId, sn_info: SnippetInfo) -> PlayerOut:
    player_in = PlayerIn(
        character_id=sn_info.character_id,
        perk_ids=sn_info.perks_ids,
        item_id=sn_info.item_id,
        addon_ids=sn_info.addons_ids,
        offering_id=sn_info.offering_id
    )
    player_out = requests.get(
        f"{URL}/form_player",
        params={
            "id": id,
            "player": player_in
        }
    )
    return PlayerOut(**player_out.json())

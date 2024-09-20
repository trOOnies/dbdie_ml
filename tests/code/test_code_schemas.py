from pytest import mark

from dbdie_ml.code.schemas import (
    check_addons_consistency,
    check_item_consistency,
    check_killer_consistency,
    check_status_consistency,
)


class MockAddon:
    def __init__(self, type_id: int) -> None:
        self.type_id = type_id


class MockPredictable:
    def __init__(self, is_for_killer: bool | None) -> None:
        self.ifk = is_for_killer


class TestCodeSchemas:
    @mark.parametrize(
        "exp,is_killer,addons_type_ids",
        [
            (True, True, [0, 0]),
            (True, True, [0, 1]),
            (True, True, [1, 0]),
            (True, True, [1, 1]),
            (True, False, [0, 0]),
            (False, False, [0, 1]),
            (False, False, [1, 0]),
            (False, False, [1, 1]),
            (True, False, [2, 0]),
            (True, False, [0, 2]),
            (False, False, [2, 1]),
            (False, False, [1, 2]),
            (True, False, [2, 2]),
            (True, False, [0, 3]),
            (True, False, [3, 0]),
            (False, False, [1, 3]),
            (False, False, [3, 1]),
            (True, False, [2, 3]),
            (True, False, [3, 2]),
            (True, False, [4, 5]),
            (True, False, [6, 7]),
        ],
    )
    def test_check_addons_consistency(
        self,
        exp,
        is_killer,
        addons_type_ids,
    ):
        addons = [MockAddon(ati) for ati in addons_type_ids]
        assert exp == check_addons_consistency(is_killer, addons)

    @mark.parametrize(
        "exp,is_killer,item_type_id",
        [
            (False, True, 0),
            (True, True, 1),
            (False, True, 2),
            (False, True, 3),
            (False, True, 4),
            (True, False, 0),
            (False, False, 1),
            (True, False, 2),
            (True, False, 3),
            (True, False, 4),
        ],
    )
    def test_check_item_consistency(self, exp, is_killer, item_type_id):
        assert exp == check_item_consistency(is_killer, item_type_id)

    @mark.parametrize(
        "exp,is_killer,is_for_killer",
        [
            (True, True, True),
            (False, True, False),
            (True, True, None),
            (False, False, True),
            (True, False, False),
            (True, False, None),
        ],
    )
    def test_check_killer_consistency(self, exp, is_killer, is_for_killer):
        predictable = MockPredictable(is_for_killer)
        assert exp == check_killer_consistency(is_killer, predictable)

    @mark.parametrize(
        "exp,status_character_id,is_killer",
        [
            (True, 0, True),
            (True, 0, False),
            (True, 1, True),
            (False, 1, False),
            (False, 2, True),
            (True, 2, False),  # ! there shouldn't be any other value
        ],
    )
    def test_check_status_consistency(
        self,
        exp,
        status_character_id,
        is_killer,
    ):
        assert exp == check_status_consistency(status_character_id, is_killer)

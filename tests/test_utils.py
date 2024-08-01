from pytest import mark, raises
from dbdie_ml.utils import filter_multitype, pls


class TestUtils:
    def test_pls(self):
        assert pls("egg", -2) == "-2 eggs"
        assert pls("egg", -1) == "-1 eggs"
        assert pls("egg", 0) == "0 eggs"
        assert pls("egg", 1) == "1 egg"
        assert pls("egg", 2) == "2 eggs"
        assert pls("egg", 3) == "3 eggs"
        assert pls("egg", 4) == "4 eggs"

    @mark.parametrize(
        "items,possible_values",
        [
            (
                "example",
                ["pv_1", "pv_2", "example", "def_1"],
            ),
            (
                ["example"],
                ["pv_1", "pv_2", "example", "def_1"],
            ),
            (
                ["example", "example"],
                ["pv_1", "pv_2", "example", "def_1"],
            ),
            (
                ["example", "pv_2"],
                ["pv_1", "pv_2", "example", "def_1"],
            ),
            (
                ["example", "pv_2", "pv_1", "def_1"],
                ["pv_1", "pv_2", "example", "def_1"],
            ),
            (
                None,
                ["pv_1", "pv_2", "example", "def_1"],
            ),
            (
                None,
                ["pv_1", "pv_2", "def_1"],
            ),
        ],
    )
    def test_filter_multitype(self, items, possible_values):
        default = ["def_1", "def_2", "def_3"]

        resp = filter_multitype(
            items,
            default,
            possible_values,
        )
        assert isinstance(resp, list)

        if items is None:
            assert resp == default
        elif isinstance(items, str):
            assert resp == [items]
        elif isinstance(items, list):
            assert resp == items
        else:
            raise TypeError("Type not possible")

    @mark.parametrize(
        "items,possible_values",
        [
            (
                [],
                ["pv_1", "pv_2", "example", "def_1"],
            ),
            (
                "example",
                ["pv_1", "pv_2", "def_1"],
            ),
            (
                ["example"],
                ["pv_1", "pv_2", "def_1"],
            ),
            (
                ["example", "pv_2", "pv_1", "def_1"],
                ["pv_1", "pv_2", "def_1"],
            ),
            (
                ["example", "pv_2", "pv_1", "def_1", "other"],
                ["pv_1", "pv_2", "example", "def_1"],
            ),
        ],
    )
    def test_filter_multitype_raises(self, items, possible_values):
        default = ["def_1", "def_2", "def_3"]
        with raises(AssertionError):
            filter_multitype(
                items,
                default,
                possible_values,
            )

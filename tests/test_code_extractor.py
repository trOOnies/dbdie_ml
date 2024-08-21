import numpy as np
from pytest import raises
from dbdie_ml.code.extractor import match_preds_types


class TestCodeExtractor:
    def test_match_preds_types(self):
        preds = np.array([1, 4, 0, 14, 4], dtype=int)
        on = "perks__killer"
        preds_out, on_out = match_preds_types(preds, on=on)
        assert preds_out == {on: preds}
        assert on_out == [on]

        preds = {
            "perks__killer": np.array([1, 4, 0, 14, 4], dtype=int),
            "perks__surv": np.array([0, 41, 2, 0, 39], dtype=int),
        }
        preds_out, on_out = match_preds_types(preds, on=None)
        assert preds_out == preds
        assert on_out == [fmt for fmt in preds.keys()]

        preds = {
            "perks__killer": np.array([1, 4, 0, 14, 4], dtype=int),
            "perks__surv": np.array([0, 41, 2, 0, 39], dtype=int),
        }
        on = "perks__killer"
        preds_out, on_out = match_preds_types(preds, on=on)
        print(preds_out)
        print(on_out)
        assert preds_out == {on: preds[on]}
        assert on_out == [on]

        preds = {
            "perks__killer": np.array([1, 4, 0, 14, 4], dtype=int),
            "perks__surv": np.array([0, 41, 2, 0, 39], dtype=int),
            "status": np.array([0, 2, 3], dtype=int),
        }
        on = ["perks__killer", "status"]
        preds_out, on_out = match_preds_types(preds, on=on)
        assert preds_out == {o: preds[o] for o in on}
        assert on_out == on

    def test_match_preds_types_raises(self):
        preds = np.array([1, 4, 0, 14, 4], dtype=int)
        with raises(AssertionError):
            match_preds_types(preds, on=None)

        preds = np.array([1, 4, 0, 14, 4], dtype=int)
        on = ["perks__killer"]
        with raises(AssertionError):
            match_preds_types(preds, on=on)

        preds = np.array([1, 4, 0, 14, 4], dtype=int)
        on = ["perks__killer", "status"]
        with raises(AssertionError):
            match_preds_types(preds, on=on)

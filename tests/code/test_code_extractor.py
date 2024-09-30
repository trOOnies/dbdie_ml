import os
from functools import partial
from shutil import rmtree

import numpy as np
from pytest import mark, raises

from dbdie_classes.version import DBDVersionRange
from backbone.code.extractor import (
    folder_save_logic,
    get_version_range,
    match_preds_types,
)


class MockModel:
    def __init__(self, selected_fd: str, version_range: DBDVersionRange):
        self.fmt = selected_fd
        self.version_range = version_range


class TestCodeExtractor:
    @mark.parametrize(
        "models,expected",
        [
            (
                {
                    "character__killer": (
                        "character__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                },
                None,
            ),
            (
                {
                    "character__killer": (
                        "character__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "character__surv": (
                        "character__surv",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "perks__killer": (
                        "perks__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                },
                None,
            ),
            (
                {
                    "character__killer": (
                        "character__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "character__surv": (
                        "character__surv",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "perks__killer": (
                        "perks__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                },
                DBDVersionRange("5.0.0", "7.0.0"),
            ),
            (
                {
                    "character__killer": (
                        "character__killer",
                        "5.0.0",
                        None,
                    ),
                    "character__surv": (
                        "character__surv",
                        "5.0.0",
                        None,
                    ),
                    "perks__killer": (
                        "perks__killer",
                        "5.0.0",
                        None,
                    ),
                },
                DBDVersionRange("5.0.0"),
            ),
        ],
    )
    def test_get_version_range(self, models, expected):
        models_proc = {
            mt: MockModel(
                vs[0],
                DBDVersionRange(vs[1], vs[2]),
            )
            for mt, vs in models.items()
        }

        dbd_vr = get_version_range(models_proc, mode="match_all", expected=expected)
        first_mt = list(models.keys())[0]
        assert dbd_vr == DBDVersionRange(
            models[first_mt][1],
            models[first_mt][2],
        )

        # For intersection changes nothing should change
        dbd_vr = get_version_range(models_proc, mode="intersection", expected=expected)
        assert dbd_vr == DBDVersionRange(
            models[first_mt][1],
            models[first_mt][2],
        )

    @mark.parametrize(
        "models,expected_ma,actual_expected_int",
        [
            (
                {
                    "perks__killer": (
                        "another_name",
                        "5.0.0",
                        "7.0.0",
                    ),
                },
                None,
                -1,
            ),
            (
                {
                    "character__killer": (
                        "character__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "character__surv": (
                        "character__surv",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "perks__killer": (
                        "perks__killer",
                        "5.0.0",
                        "7.0.1",
                    ),
                },
                None,
                DBDVersionRange("5.0.0", "7.0.0"),
            ),
            (
                {
                    "character__killer": (
                        "character__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "character__surv": (
                        "character__surv",
                        "4.0.0",
                        "7.0.0",
                    ),
                    "perks__killer": (
                        "perks__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                },
                None,
                DBDVersionRange("5.0.0", "7.0.0"),
            ),
            (
                {
                    "character__killer": (
                        "character__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "character__surv": (
                        "character__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "perks__killer": (
                        "perks__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                },
                DBDVersionRange("5.0.0"),
                -1,
            ),
            (
                {
                    "character__killer": (
                        "character__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "character__surv": (
                        "character__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                    "perks__killer": (
                        "perks__killer",
                        "5.0.0",
                        "7.0.0",
                    ),
                },
                DBDVersionRange("5.0.0", "7.0.1"),
                -1,
            ),
        ],
    )
    def test_get_version_range_raises(self, models, expected_ma, actual_expected_int):
        models_proc = {
            mt: MockModel(
                vs[0],
                DBDVersionRange(vs[1], vs[2]),
            )
            for mt, vs in models.items()
        }
        with raises(AssertionError):
            get_version_range(models_proc, mode="match_all", expected=expected_ma)

        # Intersection
        if actual_expected_int == -1:
            with raises(AssertionError):
                get_version_range(models_proc, mode="intersection")
        else:
            assert actual_expected_int == get_version_range(
                models_proc,
                mode="intersection",
            )
            assert actual_expected_int == get_version_range(
                models_proc,
                mode="intersection",
                expected=actual_expected_int,
            )

    def test_folder_save_logic(self):
        # TODO: More testing
        fsl_fd = "./tests/files/folder_save_logic"
        assert os.path.isdir(fsl_fd)
        assert os.listdir(fsl_fd) == [".gitkeep"]
        fsl = partial(os.path.join, fsl_fd)

        extr_fd = fsl("my_extractor")
        extr = partial(os.path.join, extr_fd)

        model_list = ["perks_killer", "charactersurv", "character_killer"]
        models = {
            mt: MockModel(
                mt,
                DBDVersionRange("5.0.0", "7.0.0"),
            )
            for mt in model_list
        }

        try:
            folder_save_logic(models, extr_fd, replace=False)
            assert os.path.isdir(fsl("my_extractor"))
            assert os.path.isdir(extr("models"))
            assert all(os.path.isdir(extr(f"models/{mt}")) for mt in model_list)
        finally:
            rmtree(fsl_fd)
            os.mkdir(fsl_fd)
            with open(fsl(".gitkeep"), "w") as f:
                f.write("")

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

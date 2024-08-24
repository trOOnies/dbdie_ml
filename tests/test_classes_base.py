from pytest import mark

from dbdie_ml.classes.base import CropCoords


class TestClassesBase:
    @mark.parametrize(
        "exp,crop",
        [
            (1200, (0, 10, 30, 50)),
            (11700, (0, 10, 30, 400)),
            (600, (0, 10, 15, 50)),
            (5850, (0, 10, 15, 400)),
        ],
    )
    def test_crop_coords_size(self, exp, crop):
        cc = CropCoords(*crop)
        assert exp == cc.size

    @mark.parametrize(
        "exp,c_small,c_big",
        [
            (
                True,
                (0, 0, 1920, 1080),
                (0, 0, 1920, 1080),
            ),
            (
                True,
                (1, 1, 1919, 1079),
                (0, 0, 1920, 1080),
            ),
            (
                True,
                (100, 100, 100, 100),
                (0, 0, 1920, 1080),
            ),
            (
                False,
                (0, 0, 1920, 1080),
                (100, 100, 100, 100),
            ),
            (
                False,
                (2000, 1200, 2400, 1500),
                (0, 0, 1920, 1080),
            ),
            (
                True,
                (150, 150, 250, 250),
                (100, 100, 300, 300),
            ),
            (
                False,
                (50, 150, 250, 250),  # breaks to the left
                (100, 100, 300, 300),
            ),
            (
                False,
                (150, 150, 350, 250),  # breaks to the right
                (100, 100, 300, 300),
            ),
            (
                False,
                (150, 50, 250, 250),  # breaks above
                (100, 100, 300, 300),
            ),
            (
                False,
                (150, 150, 250, 350),  # breaks below
                (100, 100, 300, 300),
            ),
        ],
    )
    def test_is_fully_inside(self, exp, c_small, c_big):
        cs = CropCoords(*c_small)
        cb = CropCoords(*c_big)
        assert exp == cs.is_fully_inside(cb)

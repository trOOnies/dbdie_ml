from dbdie_classes.extract import CropCoords
from pytest import mark, raises

from backbone.cropping.crop_settings import CropSettings


class TestCropSettings:
    @mark.parametrize(
        "crops",
        [
            [(0, 0, 1920, 1080)],
            [
                (0, 0, 960, 540),
                (960, 0, 1920, 540),
                (0, 540, 960, 1080),
                (960, 540, 1920, 1080),
            ],
        ],
    )
    def test_check_crop_shapes(self, crops):
        cs = CropSettings.from_config("img_killer_cs")

        cs.crops = {"player__killer": [CropCoords(*c) for c in crops]}
        cs.crop_shapes = {}
        cs._check_crop_shapes()

    @mark.parametrize(
        "crops",
        [
            [(0, 0, 0, 1080)],  # 0 dimension
            [(0, 0, 1920, 0)],  # 0 dimension
            [(100, 0, 100, 1080)],  # 0 dimension
            [(0, 200, 1920, 200)],  # 0 dimension
            [(1920, 0, 0, 1080)],  # negative x
            [(0, 1080, 1920, 0)],  # negative y
            [
                (0, 200, 1920, 200),
                (1920, 0, 0, 1080),
            ],  # 0 dim and negative x
            [
                (0, 0, 1920, 1080),
                (0, 0, 1920, 1080),
            ],  # same crop
            [
                (0, 0, 1920, 1080),
                (100, 200, 300, 400),
            ],  # fully inside
            [
                (100, 200, 300, 400),
                (100, 200, 300, 400),
            ],  # same crop
            [
                (100, 200, 300, 400),
                (0, 150, 150, 250),
            ],  # corner inside
            [
                (100, 200, 300, 400),
                (250, 150, 400, 250),
            ],  # same but shifted right
            [
                (100, 200, 300, 400),
                (0, 350, 150, 450),
            ],  # same but shifted down
            [
                (100, 200, 300, 400),
                (250, 350, 400, 450),
            ],  # same but 2x shifted
        ],
    )
    def test_check_crop_shapes_raises(self, crops):
        cs = CropSettings.from_config("img_surv_cs")

        cs.crops = {"player__surv": [CropCoords(*c) for c in crops]}
        cs.crop_shapes = {}
        with raises(AssertionError):
            cs._check_crop_shapes()

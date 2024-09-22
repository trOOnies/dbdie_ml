from pytest import raises

from dbdie_ml.code.cropper_swarm import cropper_fmts_nand, filter_use_croppers
from dbdie_classes.options import CROP_TYPES


class TestCodeCropperSwarm:
    def test_filter_use_croppers(self):
        cropper_flat_names = CROP_TYPES.ALL
        assert filter_use_croppers(cropper_flat_names, None) == cropper_flat_names
        assert (
            filter_use_croppers(
                cropper_flat_names,
                cropper_flat_names,
            )
            == cropper_flat_names
        )
        assert (
            filter_use_croppers(
                cropper_flat_names,
                cropper_flat_names[1:3],
            )
            == cropper_flat_names[1:3]
        )
        assert filter_use_croppers(
            cropper_flat_names,
            cropper_flat_names[3],
        ) == [cropper_flat_names[3]]

        with raises(TypeError):
            filter_use_croppers(cropper_flat_names)

    def test_cropper_fmts_nand(self):
        cropper_fmts_nand(
            use_croppers=["cp1", "cp2"],
            use_fmts=None,
        )
        cropper_fmts_nand(
            use_croppers=None,
            use_fmts=["character__killer", "points", "perks"],
        )
        cropper_fmts_nand(use_croppers=None, use_fmts=None)

        with raises(AssertionError):
            cropper_fmts_nand(
                use_croppers=["cp1", "cp2"],
                use_fmts=["character__killer", "points", "perks"],
            )

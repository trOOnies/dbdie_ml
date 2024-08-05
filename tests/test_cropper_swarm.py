from pytest import fixture
from dbdie_ml.options import CROP_TYPES
from dbdie_ml.cropper_swarm import CropperSwarm


@fixture
def mock_cropper_swarm():
    return CropperSwarm.from_types(CROP_TYPES.DEFAULT_CROP_TYPES_SEQ)


class TestCropperSwarm:
    def test_dunder_len(self, mock_cropper_swarm):
        cps: CropperSwarm = mock_cropper_swarm
        assert len(cps) == 2
        assert len(cps.cropper_flat_names) == 4

    def test_dunder_repr(self, mock_cropper_swarm):
        cps: CropperSwarm = mock_cropper_swarm
        assert str(cps) == "CropperSwarm('>=7.5.0', 2 levels, 4 croppers)"

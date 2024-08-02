from dbdie_ml.crop_settings import ALL_CS
from dbdie_ml.cropper import Cropper


class TestCropper:
    def test_instantiation(self):
        for cs in ALL_CS:
            for cpp in [Cropper(cs), Cropper.from_type(cs.name)]:
                assert cpp.settings == cs
                assert cpp.name == cs.name
                assert cpp.full_model_types == list(cs.crops.keys())
                assert cpp.full_model_types_set == set(cpp.full_model_types)

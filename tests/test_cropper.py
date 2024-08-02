import builtins
from pytest import fixture
from dbdie_ml.options import CROP_TYPES, SURV_FMT
from dbdie_ml.crop_settings import ALL_CS
from dbdie_ml.cropper import Cropper

@fixture
def mock_cropper():
    return Cropper.from_type(CROP_TYPES.SURV_PLAYER)


class TestCropper:
    def test_dunder_len(self):
        assert len(Cropper.from_type(CROP_TYPES.SURV)) == 1
        assert len(Cropper.from_type(CROP_TYPES.KILLER)) == 1
        assert len(Cropper.from_type(CROP_TYPES.SURV_PLAYER)) == 8
        assert len(Cropper.from_type(CROP_TYPES.KILLER_PLAYER)) == 8

    def test_dunder_repr(self, mock_cropper):
        cpp: Cropper = mock_cropper
        assert str(cpp) == (
            "Cropper('surv_player', 'crops/player__surv' -> 'crops/...', "
            + "version='>=7.5.0', img_size=(830, 117), 8 crops)"
        )

    def test_print_crops(self, monkeypatch, mock_cropper):
        text = []
        monkeypatch.setattr(builtins, 'print', lambda s: text.append(s))

        cpp: Cropper = mock_cropper
        try:
            cpp.print_crops()
            assert text

            i = 0
            for k, vs in cpp.settings.crops.items():
                assert text[i] == k
                i += 1

                len_vs = len(vs)
                assert text[i:i+len(vs)] == [f"- {v}" for v in vs]
                i += len_vs
        finally:
            text = []

    # * Instantiate

    def test_instantiation(self):
        for cs in ALL_CS:
            for cpp in [Cropper(cs), Cropper.from_type(cs.name)]:
                assert cpp.settings == cs
                assert cpp.name == cs.name
                assert cpp.full_model_types == list(cs.crops.keys())
                assert cpp.full_model_types_set == set(cpp.full_model_types)

    def test__filter_fmts(self, mock_cropper):
        cpp: Cropper = mock_cropper

        fmts = cpp._filter_fmts(None)
        assert fmts == SURV_FMT.ALL

        fmts = cpp._filter_fmts(SURV_FMT.OFFERING)
        assert fmts == [SURV_FMT.OFFERING]
        assert fmts[0] in SURV_FMT.ALL

        fmts = cpp._filter_fmts([SURV_FMT.OFFERING, SURV_FMT.PERKS, SURV_FMT.ITEM])
        assert fmts == [SURV_FMT.OFFERING, SURV_FMT.PERKS, SURV_FMT.ITEM]
        assert all(fmt in SURV_FMT.ALL for fmt in fmts)

    # def test_apply():  # TODO
    # def apply_from_path():  # TODO

import builtins
from pytest import fixture
from dbdie_classes.options import CROP_TYPES, SURV_FMT, KILLER_FMT
from backbone.cropping.cropper_swarm import Cropper, CropperSwarm, CropperAlignments


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

    def test_print_croppers(self, monkeypatch, mock_cropper_swarm):
        text = []
        monkeypatch.setattr(builtins, "print", lambda *s: text.append(" ".join(s)))

        cps: CropperSwarm = mock_cropper_swarm

        try:
            cps.print_croppers(verbose=False)
            assert text
            assert len(text) == 4

            assert text[0] == (
                "CROPPER SWARM: CropperSwarm('>=7.5.0', 2 levels, 4 croppers)"
            )
            assert text[1] == "CROPPERS:"
            assert text[2] == "- ['surv', 'killer']"
            assert text[3] == "- ['surv_player', 'killer_player']"
        finally:
            text = []

    def test_get_all_fmts(self, mock_cropper_swarm):
        cps: CropperSwarm = mock_cropper_swarm
        all_fmt = list(set(KILLER_FMT.ALL) | set(SURV_FMT.ALL))
        assert cps.get_all_fmts().sort() == all_fmt.sort()

    def test_cropper_alignments(self):
        cp_s = Cropper.from_type(CROP_TYPES.SURV)
        cp_k = Cropper.from_type(CROP_TYPES.KILLER)
        cp_sp = Cropper.from_type(CROP_TYPES.SURV_PLAYER)
        cp_kp = Cropper.from_type(CROP_TYPES.KILLER_PLAYER)

        assert CropperAlignments(cp_s).show_mapping() == {
            cp_s.settings.src_fd_rp: [cp_s.name]
        }
        assert CropperAlignments(cp_kp).show_mapping() == {
            cp_kp.settings.src_fd_rp: [cp_kp.name]
        }

        assert CropperAlignments([cp_s, cp_k]).show_mapping() == {
            cp_s.settings.src_fd_rp: [cp_s.name, cp_k.name]
        }
        assert CropperAlignments([cp_s, cp_sp]).show_mapping() == {
            cp_s.settings.src_fd_rp: [cp_s.name],
            cp_sp.settings.src_fd_rp: [cp_sp.name],
        }
        assert CropperAlignments([cp_sp, cp_kp]).show_mapping() == {
            cp_sp.settings.src_fd_rp: [cp_sp.name],
            cp_kp.settings.src_fd_rp: [cp_kp.name],
        }

        assert CropperAlignments([cp_s, cp_k, cp_sp, cp_kp]).show_mapping() == {
            cp_s.settings.src_fd_rp: [cp_s.name, cp_k.name],
            cp_sp.settings.src_fd_rp: [cp_sp.name],
            cp_kp.settings.src_fd_rp: [cp_kp.name],
        }

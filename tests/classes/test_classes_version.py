from pytest import mark, raises

from dbdie_ml.classes.extract import CropCoords
from dbdie_ml.classes.version import DBDVersion, DBDVersionRange
from dbdie_ml.cropping.crop_settings import CropSettings


class TestClassesVersion:
    def test_dbdversion_dunder_str(self):
        dbdv = DBDVersion("8", "1", "1")
        assert str(dbdv) == "8.1.1"
        dbdv = DBDVersion("8", "1", "1a")
        assert str(dbdv) == "8.1.1a"

    def test_dbdversion_from_str(self):
        dbvd1 = DBDVersion.from_str("8.5.0")
        assert dbvd1.major == "8"
        assert dbvd1.minor == "5"
        assert dbvd1.patch == "0"
        assert dbvd1.is_not_ptb

        dbvd2 = DBDVersion.from_str("8.5.0a")
        assert dbvd2.major == "8"
        assert dbvd2.minor == "5"
        assert dbvd2.patch == "0a"
        assert dbvd2.is_not_ptb

        dbvd3 = DBDVersion.from_str("7.2.0-ptb")
        assert dbvd3.major == "7"
        assert dbvd3.minor == "2"
        assert dbvd3.patch == "0"
        assert not dbvd3.is_not_ptb

    @mark.parametrize(
        "ineq,v1,v2",
        [
            (0, "7.5.0", "7.5.0"),
            (0, "1.5.0", "1.5.0"),
            (1, "1.5.0", "7.5.0"),
            (1, "7.5.0", "7.6.0"),
            (1, "7.5.0", "7.9.0"),
            (-1, "7.9.0", "7.5.0"),
            (1, "7.5.0", "7.5.0a"),
            (-1, "7.5.0a", "7.5.0"),
            (1, "7.5.0-ptb", "7.5.0"),
            (-1, "7.5.0", "7.5.0-ptb"),
            (-1, "8.0.0-ptb", "7.5.0"),
            (1, "7.5.0-ptb", "8.0.0"),
            (0, "7.5.0-ptb", "7.5.0-ptb"),
            (1, "7.5.0-ptb", "8.0.0-ptb"),
        ],
    )
    def test_dbdversion_dunder_ineq(self, ineq, v1, v2):
        dbdv1 = DBDVersion.from_str(v1)
        dbdv2 = DBDVersion.from_str(v2)
        if ineq == -1:
            assert dbdv1 > dbdv2
        elif ineq == 0:
            assert dbdv1 == dbdv2
        elif ineq == 1:
            assert dbdv1 < dbdv2
        else:
            raise ValueError

    def test_dbdvr_dunder_post_init(self):
        dbd_vr = DBDVersionRange("7.5.0")
        assert not dbd_vr.bounded
        assert dbd_vr._id.major == "7"
        assert str(dbd_vr._id) == "7.5.0"
        assert dbd_vr._max_id is None
        assert dbd_vr._id.is_not_ptb

        dbd_vr_2 = DBDVersionRange("7.5.0", "8.0.0")
        assert dbd_vr_2.bounded
        assert dbd_vr_2._id.major == "7"
        assert dbd_vr_2._max_id.major == "8"
        assert str(dbd_vr_2._id) == "7.5.0"
        assert str(dbd_vr_2._max_id) == "8.0.0"
        assert dbd_vr_2._id.is_not_ptb
        assert dbd_vr_2._max_id.is_not_ptb

        dbd_vr_3 = DBDVersionRange("8.0.0-ptb", "8.0.0")
        assert dbd_vr_3.bounded
        assert dbd_vr_3._id.major == "8"
        assert dbd_vr_3._max_id.major == "8"
        assert str(dbd_vr_3._id) == "8.0.0-ptb"
        assert str(dbd_vr_3._max_id) == "8.0.0"
        assert not dbd_vr_3._id.is_not_ptb
        assert dbd_vr_3._max_id.is_not_ptb

        with raises(AssertionError):
            DBDVersionRange("8.0.0", "8.0.0")
        with raises(AssertionError):
            DBDVersionRange("8.0.0", "7.5.0")
        with raises(AssertionError):
            DBDVersionRange("8.0.0", "8.0.0-ptb")

    def test_dbdvr_dunder_str(self):
        dbd_vr = DBDVersionRange("7.5.0")
        assert str(dbd_vr) == ">=7.5.0"

        dbd_vr = DBDVersionRange("7.5.0", "8.0.0")
        assert str(dbd_vr) == ">=7.5.0,<8.0.0"

    @mark.parametrize(
        "eq,v1,v1_max,v2,v2_max",
        [
            (True, "7.5.0", "8.0.0", "7.5.0", "8.0.0"),
            (True, "1.5.0", "4.5.0", "1.5.0", "4.5.0"),
            (False, "7.5.0", "8.0.0", "7.6.0", "8.0.0"),
            (False, "7.5.0", "8.0.0", "7.5.0", "9.0.0"),
            (False, "7.5.0", "8.0.0", "7.9.0", "8.1.2"),
            (False, "7.5.0", "8.0.0", "7.5.0a", "8.0.0"),
        ],
    )
    def test_dbdvr_dunder_eq(self, eq, v1, v1_max, v2, v2_max):
        dbd_vr_1 = DBDVersionRange(v1, v1_max)
        dbd_vr_2 = DBDVersionRange(v2, v2_max)
        assert (dbd_vr_1 == dbd_vr_2) == eq

    @mark.parametrize(
        "cont,cont_unbounded,v_min,v_max,v",
        [
            (False, False, "7.5.0", "8.0.0", "7.0.0"),
            (False, False, "7.5.0", "8.0.0", "7.4.9a"),
            (True, True, "7.5.0", "8.0.0", "7.5.0"),
            (True, True, "7.5.0", "8.0.0", "7.5.0a"),
            (True, True, "7.5.0", "8.0.0", "7.9.0"),
            (True, True, "7.5.0", "8.0.0", "7.9.0a"),
            (True, True, "7.5.0", "8.0.0", "7.9.9"),
            (True, True, "7.5.0", "8.0.0", "7.9.9a"),
            (False, True, "7.5.0", "8.0.0", "8.0.0"),
            (False, True, "7.5.0", "8.0.0", "8.0.0a"),
            (False, True, "7.5.0", "8.0.0", "8.0.1"),
            (False, True, "7.5.0", "8.0.0", "9.0.0"),
            (False, False, "1.5.0", "4.5.0", "1.0.0"),
            (True, True, "1.5.0", "4.5.0", "3.0.0"),
            (False, True, "1.5.0", "4.5.0", "7.5.0"),
        ],
    )
    def test_dbdvr_dunder_contains(self, cont, cont_unbounded, v_min, v_max, v):
        dbdvr = DBDVersionRange(v_min, v_max)
        dbdv = DBDVersion(*v.split("."))
        assert (dbdv in dbdvr) == cont

        dbdvr = DBDVersionRange(v_min)
        dbdv = DBDVersion(*v.split("."))
        assert (dbdv in dbdvr) == cont_unbounded

    @mark.parametrize(
        "int_min,int_max,vr1_min,vr1_max,vr2_min,vr2_max",
        [
            ("8.0.0", "8.5.0", "8.0.0", "8.5.0", "8.0.0", "8.5.0"),
            ("8.0.0", "8.5.0", "8.0.0", "9.0.0", "8.0.0", "8.5.0"),
            ("8.0.0", "8.5.0", "7.5.0", "8.5.0", "8.0.0", "8.5.0"),
            ("8.0.0", "8.5.0", "7.5.0", "9.0.0", "8.0.0", "8.5.0"),
            ("8.0.0-ptb", "8.0.0", "7.5.0", "9.0.0", "8.0.0-ptb", "8.0.0"),
            ("8.0.0-ptb", "8.0.0", "8.0.0-ptb", "8.1.0", "8.0.0-ptb", "8.0.0"),
            ("8.0.0", "8.5.0", "8.0.0", "8.5.0", "8.0.0", None),
            ("8.0.0", "8.5.0", "8.0.0", "8.5.0", "7.0.0", None),
            ("8.0.0", "8.5.0", "7.0.0", "8.5.0", "8.0.0", None),
            ("8.5.0", None, "8.0.0", None, "8.5.0", None),
            ("8.0.0", None, "8.0.0", None, "8.0.0", None),
            ("8.0.0", None, "8.0.0-ptb", None, "8.0.0", None),
        ],
    )
    def test_dbdvr_dunder_and(
        self,
        int_min,
        int_max,
        vr1_min,
        vr1_max,
        vr2_min,
        vr2_max,
    ):
        vr1 = DBDVersionRange(vr1_min, vr1_max)
        vr2 = DBDVersionRange(vr2_min, vr2_max)
        vr_int = DBDVersionRange(int_min, int_max)
        assert vr_int == vr1 & vr2
        assert vr_int == vr2 & vr1

    # def test_setup_folder  # TODO

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

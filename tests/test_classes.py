from pytest import mark
from dbdie_ml.classes import DBDVersion, DBDVersionRange


class TestClasses:
    def test_dbdversion_dunder_str(self):
        dbdv = DBDVersion("8", "1", "1")
        assert str(dbdv) == "8.1.1"
        dbdv = DBDVersion("8", "1", "1a")
        assert str(dbdv) == "8.1.1a"

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
        ],
    )
    def test_dbdversion_dunder_ineq(self, ineq, v1, v2):
        dbdv1 = DBDVersion(*v1.split("."))
        dbdv2 = DBDVersion(*v2.split("."))
        if ineq == -1:
            assert dbdv1 > dbdv2
        elif ineq == 0:
            assert dbdv1 == dbdv2
        elif ineq == 1:
            assert dbdv1 < dbdv2
        else:
            raise ValueError

    def test_dbdversionrange_dunder_post_init(self):
        dbd_vr = DBDVersionRange("7.5.0")
        assert not dbd_vr.bounded
        assert dbd_vr._id.major == "7"
        assert str(dbd_vr._id) == "7.5.0"
        assert dbd_vr._max_id is None

        dbd_vr = DBDVersionRange("7.5.0", "8.0.0")
        assert dbd_vr.bounded
        assert dbd_vr._id.major == "7"
        assert dbd_vr._max_id.major == "8"
        assert str(dbd_vr._id) == "7.5.0"
        assert str(dbd_vr._max_id) == "8.0.0"

    def test_dbdversionrange_dunder_str(self):
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
    def test_dbdversion_dunder_eq(self, eq, v1, v1_max, v2, v2_max):
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
    def test_dbdversion_dunder_contains(self, cont, cont_unbounded, v_min, v_max, v):
        dbdvr = DBDVersionRange(v_min, v_max)
        dbdv = DBDVersion(*v.split("."))
        assert (dbdv in dbdvr) == cont

        dbdvr = DBDVersionRange(v_min)
        dbdv = DBDVersion(*v.split("."))
        assert (dbdv in dbdvr) == cont_unbounded

    # def test_dunder_post_init  # TODO
    # def test_setup_folder  # TODO

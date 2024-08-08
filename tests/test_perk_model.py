from pytest import fixture
from dbdie_ml.models.custom import PerkModel


@fixture
def mock_killer_pm():
    return PerkModel(is_for_killer=True, name="mock_perks_killer")


@fixture
def mock_surv_pm():
    return PerkModel(is_for_killer=False, name="mock_perks_surv")


class TestModels:
    def test_str_like(self, mock_killer_pm):
        pm: PerkModel = mock_killer_pm
        assert not pm.str_like(None)
        assert not pm.str_like(10)
        assert not pm.str_like(0.1)
        assert not pm.str_like(True)
        assert pm.str_like("")
        assert pm.str_like("sample text")

    def test_dunder_repr(self, mock_killer_pm):
        pm: PerkModel = mock_killer_pm
        assert str(pm) == (
            "IEModel('mock_perks_killer', "
            + "type='perks', "
            + "for_killer=True, "
            + "version='>=7.5.0', "
            + "classes=None, "
            + "trained=False"
            + ")"
        )

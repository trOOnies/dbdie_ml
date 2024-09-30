from pytest import fixture

from backbone.ml.models.custom import PerkModel


@fixture
def mock_killer_pm():
    return PerkModel(is_for_killer=True, total_classes=100)


@fixture
def mock_surv_pm():
    return PerkModel(is_for_killer=False, total_classes=100)


class TestPerkModel:
    def test_dunder_repr(self, mock_killer_pm):
        pm: PerkModel = mock_killer_pm
        assert str(pm) == (
            "IEModel('md-perks-killer', "
            + "type='perks', "
            + "for_killer=True, "
            + "version='>=7.5.0', "
            + "classes=100, "
            + "trained=False"
            + ")"
        )

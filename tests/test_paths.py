from dbdie_ml.paths import absp, relp

MOCK_DBDIE_MAIN_FD = "/home/troonies/dbdie"


class TestPaths:
    def test_absp(self, monkeypatch):
        monkeypatch.setenv("DBDIE_MAIN_FD", MOCK_DBDIE_MAIN_FD)
        assert absp("data") == f"{MOCK_DBDIE_MAIN_FD}/data"
        assert absp("data/crops/status") == f"{MOCK_DBDIE_MAIN_FD}/data/crops/status"

    def test_relp(self, monkeypatch):
        monkeypatch.setenv("DBDIE_MAIN_FD", MOCK_DBDIE_MAIN_FD)
        assert relp(f"{MOCK_DBDIE_MAIN_FD}/data") == "data"
        assert relp(f"{MOCK_DBDIE_MAIN_FD}/data/crops/status") == "data/crops/status"

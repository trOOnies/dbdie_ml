from dbdie_ml.code.training import EarlyStopper


class TestModels:
    def test_early_stopper(self):
        es = EarlyStopper(patience=3)

        assert not es.early_stop(0.5000)
        assert not es.early_stop(0.4000)
        assert not es.early_stop(0.3500)
        assert not es.early_stop(0.3400)

        assert not es.early_stop(0.3500)  # 1
        assert not es.early_stop(0.3400)  # 2
        assert es.counter == 2

        assert not es.early_stop(0.3300)
        assert not es.early_stop(0.3100)
        assert not es.early_stop(0.3200)  # 1
        assert not es.early_stop(0.3300)  # 2
        assert es.early_stop(0.3200)  # 3
        assert es.counter == 3

    def test_early_stopper_with_min_delta(self):
        es = EarlyStopper(patience=3, min_delta=0.0200)

        assert not es.early_stop(0.5000)
        assert not es.early_stop(0.4000)
        assert not es.early_stop(0.3502)

        assert not es.early_stop(0.3402)  # 1
        assert not es.early_stop(0.3502)  # 2
        assert es.counter == 2

        assert not es.early_stop(0.3301)
        assert not es.early_stop(0.3001)

        assert not es.early_stop(0.2901)  # 1
        assert es.counter == 1

        assert not es.early_stop(0.2800)
        assert not es.early_stop(0.2700)  # 1
        assert not es.early_stop(0.2610)  # 2
        assert es.early_stop(0.2650)  # 3
        assert es.counter == 3

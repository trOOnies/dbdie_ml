from backbone.code.models import is_str_like


class TestCodeModels:
    def test_is_str_like(
        self,
    ):
        assert not is_str_like(None)
        assert not is_str_like(10)
        assert not is_str_like(0.1)
        assert not is_str_like(True)
        assert is_str_like("")
        assert is_str_like("sample text")

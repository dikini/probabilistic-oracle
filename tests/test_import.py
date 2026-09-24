import importlib.util


def test_package_is_importable():
    assert importlib.util.find_spec('probabilistic_oracle') is not None

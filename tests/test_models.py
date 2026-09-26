import pytest

from optionslib.models import GBMModel


def test_volatility_value_check():
    with pytest.raises(ValueError):
        GBMModel(0.0)

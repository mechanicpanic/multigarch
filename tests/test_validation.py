import numpy as np
import pytest

from multigarch import CCC, DCC, GARCH


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_garch_rejects_nonfinite(bad):
    r = np.zeros(100)
    r[5] = bad
    with pytest.raises(ValueError, match="finite"):
        GARCH().fit(r)


@pytest.mark.parametrize("cls", [CCC, DCC])
def test_multivariate_rejects_nonfinite(cls):
    r = np.zeros((100, 3))
    r[5, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cls(n_jobs=1).fit(r)


def test_garch_rejects_too_short():
    with pytest.raises(ValueError, match="observations"):
        GARCH().fit(np.array([0.1, -0.2, 0.05]))


def test_not_fitted_raises():
    with pytest.raises(ValueError, match="not fitted"):
        GARCH().forecast(5)
    with pytest.raises(ValueError, match="not fitted"):
        CCC().forecast(5)
    with pytest.raises(ValueError, match="not fitted"):
        DCC().forecast(5)

"""Keep the documentation honest: README, package guides and examples must run."""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

import multigarch
import multigarch.backtest

ROOT = Path(__file__).resolve().parents[1]


def _readme():
    path = ROOT / "README.md"
    if not path.exists():
        pytest.skip("README not shipped")
    return path.read_text()


SOURCES = {
    "README": (_readme, {}),
    "package guide": (lambda: multigarch.__doc__, {}),
    # the backtest guide uses `returns` from context
    "backtest guide": (
        lambda: multigarch.backtest.__doc__,
        {"returns": np.random.default_rng(1).standard_normal(1300) * 0.01},
    ),
}


@pytest.mark.parametrize("name", SOURCES)
def test_code_blocks_run(name, capsys):
    get_text, namespace = SOURCES[name]
    blocks = re.findall(r"```python\n(.*?)```", get_text(), re.S)
    assert blocks
    namespace = dict(namespace)
    for block in blocks:
        exec(compile(block, name, "exec"), namespace)


def test_var_backtest_example_runs(capsys):
    path = ROOT / "examples" / "var_backtest.py"
    if not path.exists():
        pytest.skip("examples not shipped")
    spec = importlib.util.spec_from_file_location("var_backtest_example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(mod.simulate(700, 3), window=500, refit_every=50)
    out = capsys.readouterr().out
    assert "Kupiec" in out and "GJR-t" in out

# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/multigarch`: `models.py` exposes `GARCH`, `CCC`, `DCC`; `_jit.py` holds NumPy/numba-accelerated inner loops; `__init__.py` wires exports and version.
- Keep new utilities close to their model (helpers in `models.py`, performance-critical pieces in `_jit.py` with `@njit(cache=True)`).
- Place tests under `tests/` mirroring the `src/` layout (e.g., `tests/test_models.py`) and use realistic array shapes (`(T, n)` returns).

## Build, Test, and Development Commands
- Install dev deps: `python -m pip install -e .[dev]`.
- Run tests: `python -m pytest` (supports standard `pytest` flags).
- Build wheel/sdist: `python -m pip install build` then `python -m build` (uses Hatchling backend defined in `pyproject.toml`).
- Linting/formatting are not enforced by tooling; favor PEP 8, type hints, and consistent docstrings. If you add a linter, prefer `ruff` or `flake8` with minimal config.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; type annotate public APIs (`NDArray[np.float64]`, etc.).
- Classes in `PascalCase`, functions/variables in `snake_case`; prefix internal helpers with `_`.
- Keep numerical code allocation-light; reuse arrays where possible and guard against negative variances or unstable persistence (>1).
- Document method behavior and expected shapes; keep examples small and reproducible.

## Testing Guidelines
- Use `pytest` with `test_*.py` files and `test_*` functions; parametrize across orders `(p, q)` and asset counts.
- Seed randomness when generating synthetic returns and assert both shape and stability (e.g., variances stay positive, correlations symmetric).
- For numba paths, add small sanity checks rather than large golden outputs to avoid brittle tests.

## Commit & Pull Request Guidelines
- Recent history favors concise, imperative summaries (`Add low_memory mode…`); follow that style, keep subject lines ≤72 chars, and scope each commit to one logical change.
- PRs should describe the motivation, approach, and risk areas; list commands/tests run; link issues if any. Include performance notes when changing JIT loops or parallelism (`n_jobs`).
- Prefer small, reviewable diffs; highlight backward-incompatible changes to APIs or defaults.

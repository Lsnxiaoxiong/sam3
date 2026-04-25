# Repository Guidelines

## Project Structure & Module Organization
`sam3/` contains the main Python package. Keep reusable production code there: model code in `sam3/model/`, training logic in `sam3/train/`, evaluation helpers in `sam3/eval/`, and agent utilities in `sam3/agent/`. Use `scripts/` for one-off utilities, `examples/` for notebooks, `docs/` for workflow notes, and `assets/` for demo media. Generated outputs belong in `output/`, `build/`, or `dist/`, not inside package modules.

## Build, Test, and Development Commands
Use editable installs during development:

```bash
pip install -e .
pip install -e ".[dev,train]"
pip install -e ".[notebooks]"
```

`pip install -e .` installs the package, `.[dev,train]` adds testing, formatting, and training dependencies, and `.[notebooks]` adds notebook extras.

Format code with:

```bash
ufmt format .
```

Run tests with:

```bash
pytest sam3/perflib/tests/tests.py
```

`pyproject.toml` points `pytest` at `tests/`, but the tracked test file currently lives under `sam3/perflib/tests/`. Use the explicit path until that mismatch is resolved.

Training examples are documented in `README_TRAIN.md`, for example:

```bash
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml
```

## Coding Style & Naming Conventions
Follow Black-compatible Python style: 4-space indentation and an 88-character line length. Prefer clear type-aware signatures where practical. Use `snake_case` for modules, files, and functions, `PascalCase` for classes, and descriptive dataset-oriented names for config files such as `odinw_text_only_train.yaml`. Keep top-level scripts narrow and move reusable logic into `sam3/`.

## Testing Guidelines
Pytest is the configured test runner. Name new test files `test_*.py` and test functions `test_*`. Add focused regression coverage for behavior you change, especially around tensor shapes, mask outputs, config loading, and inference utilities.

## Commit & Pull Request Guidelines
Recent history favors short imperative subjects such as `feat: ...`, `feat(model): ...`, and `chore: ...`. Keep commits scoped and descriptive. For pull requests, branch from `main`, summarize behavior changes, link related issues, update docs when workflows or APIs change, and include screenshots or sample outputs for notebook, visualization, or inference changes.

## Security & Configuration Tips
Do not commit checkpoints, secrets, or machine-specific dataset paths. Keep Hugging Face credentials and large datasets outside the repository, and pass local paths through configs or environment-specific overrides.

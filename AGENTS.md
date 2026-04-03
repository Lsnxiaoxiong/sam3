# Repository Guidelines

## Project Structure & Module Organization
`sam3/` contains the Python package and most production code. Core model code lives in `sam3/model/`, training code in `sam3/train/`, evaluation utilities in `sam3/eval/`, and agent helpers in `sam3/agent/`. Use `scripts/` for one-off data prep and evaluation helpers, `examples/` for Jupyter notebooks, `docs/` for workflow notes, and `assets/` for demo media. Generated artifacts belong in `output/`, `build/`, or `dist/` and should not be mixed into package modules.

## Build, Test, and Development Commands
Use editable installs while developing:

```bash
pip install -e .
pip install -e ".[dev,train]"
pip install -e ".[notebooks]"
```

`pip install -e .` installs the package, `.[dev,train]` adds formatting, testing, and training dependencies, and `.[notebooks]` adds notebook extras. Format code with:

```bash
ufmt format .
```

Run tests with `pytest`. Note that `pyproject.toml` points to `tests/`, but the tracked test file currently sits at `sam3/perflib/tests/tests.py`, so use:

```bash
pytest sam3/perflib/tests/tests.py
```

Training entrypoint examples are in `README_TRAIN.md`, for example:

```bash
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml
```

## Coding Style & Naming Conventions
Follow Black-compatible Python style: 4-space indentation, 88-character lines, and clear type-aware function signatures where practical. Module and file names use `snake_case`; classes use `PascalCase`; config files under `sam3/train/configs/` use descriptive dataset-oriented names such as `odinw_text_only_train.yaml`. Keep new scripts narrow in scope and place reusable logic inside `sam3/`, not top-level files.

## Testing Guidelines
Pytest is the configured test runner. Name new tests `test_*.py` and test functions `test_*`. Prefer colocated focused tests for low-level utilities and add regression coverage for changed behavior, especially around tensor shapes, mask outputs, and config loading.

## Commit & Pull Request Guidelines
Recent history uses short conventional-style subjects such as `feat: ...`, `feat(examples): ...`, and `chore: ...`. Keep the subject imperative and under a single line. For pull requests, branch from `main`, describe behavior changes, link issues when relevant, update docs for API or workflow changes, and include screenshots or sample outputs for notebook, visualization, or inference changes. Complete the Facebook CLA before requesting review.

## Security & Configuration Tips
Do not commit checkpoints, secrets, or local dataset paths. Keep Hugging Face authentication and large training data outside the repository, and pass machine-specific paths through configs or environment-specific overrides.

---
trigger: always_on
description: Strict environment and dependency rules for PyKoopman development (Python 3.11, uv, pinned numpy/sklearn).
---

# PyKoopman Project Rules

## 1. Environment Management
- **Tooling**: ALWAYS use **`uv`** for Python environment management.
- **Python Version**: Target **Python 3.11**.
  - `uv venv --python 3.11`
  - `uv pip install -e .`

## 2. Dependency Management (CRITICAL)
- **Strict Pinning**:
  - `numpy`: **`>= 1.20, <= 1.26`** (STRICTLY 1.x series. Numpy 2.0+ is incompatible).
  - `scikit-learn`: **`== 1.1.3`** (Strict pin required).
  - `scipy`: **`> 1.6.0, <= 1.11.2`**.
  - `pydmd`: **`> 0.4, <= 0.4.1`**.
- **GPU Support**: Use `uv` PyTorch integration guide. Do not rely on default resolution.

## 3. Testing
- Unit Tests: `uv run pytest`
- Notebooks: `uv run papermill docs/notebook.ipynb nul` (Avoid `jupyter_contrib_nbextensions`).

## 4. Configuration Consistency
- Updates to Python version must be applied atomically across:
  - `pyproject.toml`
  - `README.rst`
  - `.readthedocs.yaml`
  - `.github/workflows/*.yml`

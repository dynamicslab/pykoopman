# PyKoopman v1.2.0 Release Notes

## 🚀 Major Updates: Stable Environment & Dependency Fixes

This release focuses on establishing a rock-solid, reproducible development environment and resolving "dependency hell" issues.

### 🐍 Environment Standardization
- **Python 3.11** is now the official target version (upgraded from 3.10).
- **`uv` Adoption**: The project now fully recommends and supports `uv` for lightning-fast environment management.
- **Workflow Simplified**: `[dev]` dependencies have been merged into the main package. Development installation is now simply:
  ```bash
  uv pip install -e .
  ```

### 📦 Dependency Resolution
- **Strict Pinning**: fixed critical version mismatches causing installation failures:
  - `numpy`: Pinned to `1.x` series (invalidated `2.0+` which breaks `pydmd`).
  - `scikit-learn`: Pinned to `1.1.3` to match `pykoopman` API usage.
  - `scipy` & `pydmd`: Explicitly bounded to known working versions.
- **Cleanup**: Legacy `requirements.txt` files have been archived to `_old/`.

### 🛠️ Configuration & Quality
- **CI/CD**: GitHub Actions (`run-tests.yml`, `release.yml`) and ReadTheDocs updated to Python 3.11.
- **Project Structure**: `pyproject.toml` modernized and cleaned up.
- **Documentation**: `README.rst` updated with clear, modern installation instructions.

### 🧪 Verification
- Verified clean installation on new environments.
- **100% Pass Rate**: All 1709 tests passing.
- Jupyter Notebooks verified via `papermill`.

---
*Release drafted from branch `fix-env-issues`*

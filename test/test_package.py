"""Package-level checks: importing pykoopman and its declared dependencies."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import tomllib
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# setuptools 82 removed pkg_resources. lightning 2.0 to 2.3 still call
# __import__("pkg_resources").declare_namespace(...) in
# lightning/fabric/__init__.py and lightning/pytorch/__init__.py; 2.4.0 does not.
LIGHTNING_NEEDS_PKG_RESOURCES = SpecifierSet(">=2.0, <2.4")
# Probe versions covering every final and post release of lightning 2.x.
LIGHTNING_PROBES = [
    f"2.{minor}.{patch}{post}"
    for minor in range(10)
    for patch in range(10)
    for post in ("", ".post0")
]


def test_import_does_not_need_pkg_resources():
    # Import pykoopman in a fresh interpreter with pkg_resources blocked. While
    # an affected lightning release is installed it is imported before the
    # block, so the check covers pykoopman and everything else it pulls in.
    preimport = ""
    if version("lightning") in LIGHTNING_NEEDS_PKG_RESOURCES:
        preimport = "import lightning"
    code = textwrap.dedent(
        f"""
        import sys
        {preimport}
        sys.modules["pkg_resources"] = None
        import pykoopman
        print(pykoopman.__version__)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, MPLBACKEND="Agg"),
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == version("pykoopman")


def test_setuptools_capped_while_lightning_needs_pkg_resources():
    # A fresh resolve otherwise picks the newest setuptools, and with
    # setuptools >= 82 an affected lightning release fails at import.
    with PYPROJECT.open("rb") as f:
        dependencies = tomllib.load(f)["project"]["dependencies"]
    requirements = {}
    for line in dependencies:
        requirement = Requirement(line)
        requirements[canonicalize_name(requirement.name)] = requirement

    lightning = requirements["lightning"].specifier
    affected = [v for v in LIGHTNING_PROBES if v in LIGHTNING_NEEDS_PKG_RESOURCES]
    if any(lightning.contains(v) for v in affected):
        setuptools = requirements.get("setuptools")
        assert setuptools is not None, "lightning < 2.4 needs setuptools < 82"
        assert setuptools.marker is None
        for v in ("82.0.0", "84.2.1", "100.0.0"):
            assert not setuptools.specifier.contains(v), v


# Test, documentation, lint and notebook tools. As runtime requirements they
# would pin the pytest, Sphinx and Jupyter of every environment that installs
# pykoopman, so they belong in the extras under [project.optional-dependencies].
TOOLING = frozenset(
    canonicalize_name(name)
    for name in (
        "black",
        "flake8",
        "flake8-builtins-unleashed",
        "ipykernel",
        "ipython",
        "ipywidgets",
        "jupyter",
        "jupyterlab",
        "nbsphinx",
        "notebook",
        "pre-commit",
        "pytest",
        "pytest-cov",
        "pytest-lazy-fixture",
        "pytest-lazy-fixtures",
        "setuptools-scm",
        "setuptools-scm-git-archive",
        "sphinx",
        "sphinx-codeautolink",
        "sphinx-rtd-theme",
        "sphinxcontrib-apidoc",
    )
)


def _project_table():
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]


def _names(lines):
    return {canonicalize_name(Requirement(line).name) for line in lines}


def test_runtime_dependencies_exclude_tooling():
    misplaced = _names(_project_table()["dependencies"]) & TOOLING
    assert not misplaced, f"move to an extra: {sorted(misplaced)}"


def test_extras_provide_test_and_docs_tooling():
    extras = _project_table().get("optional-dependencies", {})
    # CI installs .[test] and .readthedocs.yaml installs .[docs].
    test = _names(extras.get("test", []))
    docs = _names(extras.get("docs", []))
    assert {"pytest", "pytest-cov", "pytest-lazy-fixtures"} <= test
    assert {"sphinx", "nbsphinx", "sphinx-codeautolink", "sphinxcontrib-apidoc"} <= docs
    assert "sphinx-rtd-theme" in docs
    # nbsphinx needs IPython's ipython3 lexer and ipywidgets for the stored
    # widget outputs of the tutorial notebooks.
    assert {"ipython", "ipywidgets"} <= docs
    # pytest-lazy-fixture breaks test collection on pytest 8 and newer.
    for extra, lines in extras.items():
        assert "pytest-lazy-fixture" not in _names(lines), extra

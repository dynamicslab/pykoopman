PyKoopman
=========

|Build| |Docs| |PyPI| |Codecov| |DOI| |JOSS|

**PyKoopman** is a Python package for computing data-driven approximations to the Koopman operator.

Data-driven approximation of Koopman operator
---------------------------------------------

.. figure:: docs/JOSS/Fig1.png

Given a nonlinear dynamical system,

.. math::

   x'(t) = f(x(t)),

the Koopman operator governs the temporal evolution of the measurement function.
Unfortunately, it is an infinite-dimensional linear operator. Most of the time, one has to
project the Koopman operator onto a finite-dimensional subspace that is spanned by user-defined/data-adaptive functions.

.. math::
    z = \Phi(x).

If the system state is also contained in such subspace, then effectively, the nonlinear dynamical system is (approximately)
linearized in a global sense.

The goal of data-driven approximation of Koopman operator is to find such a set of
functions that span such lifted space and the transition matrix associated with the
lifted system.

Structure of PyKoopman
^^^^^^^^^^^^^^^^^^^^^^

.. figure:: docs/JOSS/Fig2.png

PyKoopman package is centered around the ``Koopman`` class and ``KoopmanContinuous`` class. It consists of two key components

* ``observables``: a set of observables functions, which spans the subspace for projection.

* ``regressor``: the optimization algorithm to find the best ``fit`` for the
  projection of Koopman operator.

After ``Koopman``/``KoopmanContinuous`` object has been created, it must be fit to data, similar to a ``scikit-learn`` model.
We design ``PyKoopman`` such that it is compatible to ``scikit-learn`` objects and methods as much as possible.


Features implemented
^^^^^^^^^^^^^^^^^^^^

- Observable library for lifting the state into the observable space

  - Identity (for DMD/DMDc or in case users want to compute observables themselves):
    ``Identity``
  - Multivariate polynomials: ``Polynomial``
  - Time delay coordinates: ``TimeDelay``
  - Radial basis functions: ``RadialBasisFunctions``
  - Random Fourier features: ``RandomFourierFeatures``
  - Custom library (defined by user-supplied functions): ``CustomObservables``
  - Concatenation of observables: ``ConcatObservables``


- System identification method for performing regression

  - Dynamic mode decomposition: ``PyDMDRegressor``
  - Dynamic mode decomposition with control: ``DMDc``
  - Extended dynamic mode decomposition: ``EDMD``
  - Extended dynamic mode decomposition with control: ``EDMDc``
  - Kernel dynamic mode decomposition: ``KDMD``
  - Hankel Alternative View of Koopman Analysis: ``HAVOK``
  - Neural Network DMD: ``NNDMD``

- Sparse construction of Koopman invariant subspace

  - Multi-task learning based on linearity consistency


Examples
^^^^^^^^

1. `Learning how to create observables <https://pykoopman.readthedocs.io/en/master/tutorial_compose_observables.html>`__

2. `Learning how to compute time derivatives <https://pykoopman.readthedocs.io/en/master/tutorial_compute_differentiation.html>`__

3. `Dynamic mode decomposition on two mixed spatial signals <https://pykoopman.readthedocs.io/en/master/tutorial_dmd_separating_two_mixed_signals_400d_system.html>`__

4. `Dynamic mode decomposition with control on a 2D linear system <https://pykoopman.readthedocs.io/en/master/tutorial_dmd_with_control_2d_system.html>`__

5. `Dynamic mode decomposition with control (DMDc) for a 128D system <https://pykoopman.readthedocs.io/en/master/tutorial_dmd_with_control_128d_system.html>`__

6. `Dynamic mode decomposition with control on a high-dimensional linear system <https://pykoopman.readthedocs.io/en/master/tutorial_linear_random_control_system.html>`__

7. `Successful examples of using Dynamic mode decomposition on PDE system <https://pykoopman.readthedocs.io/en/master/tutorial_dmd_succeeds_pde_examples.html>`__

8. `Unsuccessful examples of using Dynamic mode decomposition on PDE system <https://pykoopman.readthedocs.io/en/master/tutorial_dmd_failed_for_pde_examples.html>`__

9. `Extended DMD for Van Der Pol System <https://pykoopman.readthedocs.io/en/master/tutorial_koopman_edmd_with_rbf.html>`__

10. `Learning Koopman eigenfunctions on Slow manifold <https://pykoopman.readthedocs.io/en/master/tutorial_koopman_eigenfunction_model_slow_manifold.html>`__

11. `Comparing DMD and KDMD for Slow manifold dynamics <https://pykoopman.readthedocs.io/en/master/tutorial_koopman_kdmd_on_slow_manifold.html>`__

12. `Extended DMD with control for chaotic duffing oscillator <https://pykoopman.readthedocs.io/en/master/tutorial_koopman_edmdc_for_chaotic_duffing_oscillator.html>`__

13. `Extended DMD with control for Van der Pol oscillator <https://pykoopman.readthedocs.io/en/master/tutorial_koopman_edmdc_for_vdp_system.html>`__

14. `Hankel Alternative View of Koopman Operator for Lorenz System <https://pykoopman.readthedocs.io/en/master/tutorial_koopman_havok_3d_lorenz.html>`__

15. `Hankel DMD with control for Van der Pol Oscillator <https://pykoopman.readthedocs.io/en/master/tutorial_koopman_hankel_dmdc_for_vdp_system.html>`__

16. `Neural Network DMD on Slow Manifold <https://pykoopman.readthedocs.io/en/master/tutorial_koopman_nndmd_examples.html>`__

17. `EDMD and NNDMD for a simple linear system <https://pykoopman.readthedocs.io/en/master/tutorial_linear_system_koopman_eigenfunctions_with_edmd_and_nndmd.html>`__

18. `Sparisfying a minimal Koopman invariant subspace from EDMD for a simple linear system <https://pykoopman.readthedocs.io/en/master/tutorial_sparse_modes_selection_2d_linear_system.html>`__

Installation
-------------

Language
^^^^^^^^^^^^^^^^^^^^
- Python == 3.10


Installing with pip
^^^^^^^^^^^^^^^^^^^

If you are using Linux or macOS you can install PyKoopman with pip:

.. code-block:: bash

  pip install pykoopman

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
First clone this repository:

.. code-block:: bash

  git clone https://github.com/dynamicslab/pykoopman

Second, it is highly recommended to use `venv` to get a local python environment

.. code-block:: bash

  python -m venv venv
  source ./venv/bin/activate

In windows, you activate virtual environment in a different way

.. code-block:: bash

  .\venv\Scripts\activate.ps1

Then, to install the package, run

.. code-block:: bash

  python -m pip install -e .

If you do not have root access, you should add the ``--user`` option to the above lines.


Installing with GPU support
^^^^^^^^^^^^^^^^^^^^^^^^^^^

After you download the Github package, go to the directory, type

.. code-block:: bash

  python -m pip install -r requirements-dev.txt

Documentation
-------------
The documentation for PyKoopman is hosted on `Read the Docs <https://pykoopman.readthedocs.io/en/latest/>`__.

Community guidelines
--------------------

Contributing code
^^^^^^^^^^^^^^^^^
We welcome contributions to PyKoopman. To contribute a new feature please submit a
pull request. To get started we recommend installing the packages in "developer mode"
via

.. code-block:: bash

    python -m pip install -e .[dev]

This will allow you to run unit tests and automatically format your code. To be accepted your code should conform to PEP8 and pass all unit tests. Code can be tested by invoking

.. code-block:: bash

    pytest

We recommed using ``pre-commit`` to format your code. Once you have staged changes to commit

.. code-block:: bash

    git add path/to/changed/file.py

you can run the following to automatically reformat your staged code

.. code-block:: bash

    pre-commit -a -v

Note that you will then need to re-stage any changes ``pre-commit`` made to your code.

Reporting issues or bugs
^^^^^^^^^^^^^^^^^^^^^^^^
If you find a bug in the code or want to request a new feature, please open an issue.

Known issues:

- Python 3.12 might cause unexpected problems.

Citing PyKoopman
----------------

.. code-block:: text

    @article{Pan2024, doi = {10.21105/joss.05881},
    url = {https://doi.org/10.21105/joss.05881},
    year = {2024},
    publisher = {The Open Journal},
    volume = {9},
    number = {94},
    pages = {5881},
    author = {Shaowu Pan and Eurika Kaiser and Brian M. de Silva and J. Nathan Kutz and Steven L. Brunton},
    title = {PyKoopman: A Python Package for Data-Driven Approximation of the Koopman Operator},
    journal = {Journal of Open Source Software}}

Related packages
----------------
* `PySINDy <https://github.com/dynamicslab/pysindy/>`_ - A Python libray for the Sparse Identification of Nonlinear Dynamical
  systems (SINDy) method introduced in Brunton et al. (2016a).
* `Deeptime <https://github.com/deeptime-ml/deeptime>`_ - A Python library for the analysis of time series data with methods for dimension reduction, clustering, and Markov model estimation.
* `PyDMD <https://github.com/mathLab/PyDMD/>`_ - A Python package using the Dynamic Mode Decomposition (DMD) for a data-driven model simplification based on spatiotemporal coherent structures. DMD is a great alternative to SINDy.
* `pykoop <https://github.com/decargroup/pykoop>`_ - a Koopman operator identification library written in Python
* `DLKoopman <https://github.com/GaloisInc/dlkoopman>`_ - a deep learning library for
  Koopman operator

.. |Build| image:: https://github.com/dynamicslab/pykoopman/actions/workflows/run-tests.yml/badge.svg
    :target: https://github.com/dynamicslab/pykoopman/actions?query=workflow%3ATests

.. |Docs| image:: https://readthedocs.org/projects/pykoopman/badge/?version=master
    :target: https://pykoopman.readthedocs.io/en/master/?badge=master
    :alt: Documentation Status

.. |PyPI| image:: https://badge.fury.io/py/pykoopman.svg
    :target: https://badge.fury.io/py/pykoopman

.. |Codecov| image:: https://codecov.io/github/dynamicslab/pykoopman/coverage.svg
    :target: https://app.codecov.io/gh/dynamicslab/pykoopman

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8060893.svg
    :target: https://doi.org/10.5281/zenodo.8060893

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.05881/status.svg
    :target: https://doi.org/10.21105/joss.05881

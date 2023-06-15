PyKoopman
=========

|Build| |Docs| |PyPI| |Codecov|

**PyKoopman** is a Python package for computing data-driven approximations to the Koopman operator.

Data-driven approximation of Koopman operator
---------------------------------------------

Given a nonlinear dynamical system,

.. math::

   x'(t) = f(x(t)),

the Koopman operator governs the temporal evolution of the measurement function.
Unfortunately, it is an infinite-dimensional linear operator. Most of the time, one has to
project the Koopman operator onto a finite-dimensional subspace that is spanned by user-defined/data-adaptive functions.
If the system state is also contained in such subspace, then effectively, the nonlinear dynamical system is (approximately)
linearized in a global sense.

Structure of PyKoopman
^^^^^^^^^^^^^^^^^^^^^^
PyKoopman package is centered around the ``Koopman`` class and ``KoopmanContinuous`` class. It consists of two key components

* ``observables``: a set of observables functions, which spans the subspace for projection.

* ``regressor``: the optimization algorithm to find the best `fit` for the projection of Koopman operator.

After ``Koopman``/``KoopmanContinuous`` object has been created, it must be fit to data, similar to a ``scikit-learn`` model.
We design ``PyKoopman`` such that it is compatible to ``scikit-learn`` objects and methods as much as possible.


Example
^^^^^^^

1. `Learning how to create observables <https://pykoopman.readthedocs
.io/en/master/tutorial_compose_observables
.html>`__

2. `Learning how to compute time derivatives <https://pykoopman.readthedocs
.io/en/master/tutorial_compute_differentiation.html>`__

3. `Dynamic mode decomposition on two mixed spatial signals <https://pykoopman.readthedocs.io/en/master/tutorial_dmd_separating_two_mixed_signals_400d_system.html>`__

4. `Dynamic mode decomposition with control on a 2D linear system <https://pykoopman
.readthedocs.io/en/master/tutorial_dmd_with_control_2d_system
.html>`__

5. `Dynamic mode decomposition with control (DMDc) for a 128D system <https://pykoopman.readthedocs.io/en/master/tutorial_dmd_with_control_128d_system.html>`__

6. `Dynamic mode decomposition with control on a high-dimensional linear system
<https://pykoopman.readthedocs.io/en/master/tutorial_linear_random_control_system
.html>`__

7. `Successful examples of using Dynamic mode decomposition on PDE system
<https://pykoopman.readthedocs.io/en/master/tutorial_dmd_succeeds_pde_examples
.html>`__

8. `Unsuccessful examples of using Dynamic mode decomposition on PDE system <https://pykoopman.readthedocs.io/en/master/tutorial_dmd_failed_for_pde_examples.html>`__

9. `Extended DMD for Van Der Pol System <https://pykoopman.readthedocs
.io/en/master/tutorial_koopman_edmd_with_rbf.html>`__

10. `Learning Koopman eigenfunctions on Slow manifold <https://pykoopman.readthedocs
.io/en/master/tutorial_koopman_eigenfunction_model_slow_manifold.html>`__

11. `Comparing DMD and KDMD for Slow manifold dynamics <https://pykoopman.readthedocs
.io/en/master/tutorial_koopman_kdmd_on_slow_manifold.html>`__

12. `Extended DMD with control for chaotic duffing oscillator <https://pykoopman.readthedocs.io/en/master/tutorial_koopman_edmdc_for_chaotic_duffing_oscillator.html>`__

13. `Extended DMD with control for Van der Pol oscillator <https://pykoopman.readthedocs
.io/en/master/tutorial_koopman_edmdc_for_vdp_system.html>`__

14. `Hankel Alternative View of Koopman Operator for Lorenz System <https://pykoopman.readthedocs.io/en/master/tutorial_koopman_havok_3d_lorenz.html>`__

15. `Hankel DMD with control for Van der Pol Oscillator <https://pykoopman.readthedocs
.io/en/master/tutorial_koopman_hankel_dmdc_for_vdp_system.html>`__

16. `Neural Network DMD on Slow Manifold <https://pykoopman.readthedocs
.io/en/master/tutorial_koopman_nndmd_examples
.html>`__

17. `EDMD and NNDMD for a simple linear system <https://pykoopman.readthedocs
.io/en/master/tutorial_linear_system_koopman_eigenfunctions_with_edmd_and_nndmd.html>`__

18. `Sparisfying a minimal Koopman invariant subspace from EDMD for a simple linear
system <https://pykoopman.readthedocs
.io/en/master/tutorial_sparse_modes_selection_2d_linear_system.html>`__

Installation
-------------

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

Then, to install the package, run

.. code-block:: bash

  pip install .

If you do not have pip you can instead use

.. code-block:: bash

  python setup.py install

If you do not have root access, you should add the ``--user`` option to the above lines.

Documentation
-------------
The documentation for PyKoopman is hosted on `Read the Docs <https://pykoopman.readthedocs.io/en/latest/>`__.

Community guidelines
--------------------

Contributing code
^^^^^^^^^^^^^^^^^
We welcome contributions to PyKoopman. To contribute a new feature please submit a pull request. To get started we recommend installing the packages in ``requirements-dev.txt`` via

.. code-block:: bash

    pip install -r requirements-dev.txt

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

References
------------

-  Williams, Matthew O., Ioannis G. Kevrekidis, and Clarence W. Rowley.
   *A data–driven approximation of the koopman operator: Extending dynamic mode
   decomposition.* Journal of Nonlinear Science 25, no. 6 (2015): 1307-1346.
   `[DOI] <https://doi.org/10.1007/s00332-015-9258-5>`_

-  Williams, Matthew O., Clarence W. Rowley, and Ioannis G. Kevrekidis.
   *A kernel-based approach to data-driven Koopman spectral analysis.* arXiv
   preprint arXiv:1411.2260 (2014).
   `[DOI] <https://doi.org/10.48550/arXiv.1411.2260>`_

-  Brunton, Steven L., et al. *Chaos as an intermittently forced linear system.*
   Nature communications 8.1 (2017): 1-9.
   `[DOI] <https://doi.org/10.1038/s41467-017-00030-8>`_

-  Kaiser, Eurika, J. Nathan Kutz, and Steven L. Brunton.
   *Data-driven discovery of Koopman eigenfunctions for control.*
   Machine Learning: Science and Technology 2.3 (2021): 035023.
   `[DOI] <https://doi.org/10.1088/2632-2153/abf0f5>`_

-  Lusch, Bethany, J. Nathan Kutz, and Steven L. Brunton.
   *Deep learning for universal linear embeddings of nonlinear dynamics.* Nature
   communications 9.1 (2018): 4950.
   `[DOI] <https://doi.org/10.1038/s41467-018-07210-0>`_

-  Otto, Samuel E., and Clarence W. Rowley. *Linearly recurrent autoencoder networks
   for learning dynamics.* SIAM Journal on Applied Dynamical Systems 18.1 (2019):
   558-593.
   `[DOI] <https://doi.org/10.1137/18M1177846>`_

-  Pan, Shaowu, Nicholas Arnold-Medabalimi, and Karthik Duraisamy.
   *Sparsity-promoting algorithms for the discovery of informative Koopman-invariant
   subspaces.* Journal of Fluid Mechanics 917 (2021).
   `[DOI] <https://doi.org/10.1017/jfm.2021.271>`_


.. |Build| image:: https://github.com/dynamicslab/pykoopman/workflows/Tests/badge.svg
    :target: https://github.com/dynamicslab/pykoopman/actions?query=workflow%3ATests

.. |Docs| image:: https://readthedocs.org/projects/pykoopman/badge/?version=latest
    :target: https://pykoopman.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |PyPI| image:: https://badge.fury.io/py/pykoopman.svg
    :target: https://badge.fury.io/py/pykoopman

.. |Codecov| image:: https://codecov.io/github/dynamicslab/pykoopman/coverage.svg
    :target: https://app.codecov.io/gh/dynamicslab/pykoopman

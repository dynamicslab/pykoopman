"""
Polynomial observables
"""
from itertools import chain
from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
from scipy import sparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing._csr_polynomial_expansion import _csr_polynomial_expansion
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES

from ..common import check_array


class Polynomial(PolynomialFeatures):
    """Polynomial observables.

    This is essentially just :code:`sklearn.preprocessing.PolynomialFeatures`
    with support for complex numbers.

    Parameters
    ----------
    degree : int, optional (default 2)
        The degree of the polynomial features.
    interaction_only : boolean, optional (default False)
        If true, only interaction features are produced: features that are
        products of at most ``degree`` *distinct* input features (so not
        ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).
    include_bias : boolean, optional (default True)
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
    order : str in {'C', 'F'}, optional (default 'C')
        Order of output array in the dense case. 'F' order is faster to
        compute, but may slow down subsequent estimators.
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True, order="C"):
        if degree == 0:
            raise ValueError(
                "degree must be at least 1, otherwise inverse cannot be computed"
            )
        super(Polynomial, self).__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order=order,
        )

    def fit(self, x, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
            Measurement data.

        y : array-like, optional (default None)
            Dummy input.
        Returns
        -------
        self : fit :class:`pykoopman.observables.Polynomial` instance
        """
        return super(Polynomial, self).fit(x.real, y)

    def transform(self, x):
        """Transform data to polynomial features.

        Parameters
        ----------
        x : array-like or CSR/CSC sparse matrix, shape (n_samples, n_features)
            The data to transform, row by row.
            Prefer CSR over CSC for sparse input (for speed), but CSC is
            required if the degree is 4 or higher. If the degree is less than
            4 and the input format is CSC, it will be converted to CSR, have
            its polynomial features generated, then converted back to CSC.
            If the degree is 2 or 3, the method described in "Leveraging
            Sparsity to Speed Up Polynomial Feature Expansions of CSR Matrices
            Using K-Simplex Numbers" by Andrew Nystrom and John Hughes is
            used, which is much faster than the method used on CSC input. For
            this reason, a CSC input will be converted to CSR, and the output
            will be converted back to CSC prior to being returned, hence the
            preference of CSR.

        Returns
        -------
        xp : np.ndarray or CSR/CSC sparse matrix, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of polynomial
            features generated from the combination of inputs.
        """
        check_is_fitted(self, "n_input_features_")

        x = check_array(x, order="F", dtype=FLOAT_DTYPES, accept_sparse=("csr", "csc"))

        n_samples, n_features = x.shape

        if n_features != self.n_input_features_:
            raise ValueError("x shape does not match training shape")

        if sparse.isspmatrix_csr(x):
            if self.degree > 3:
                return self.transform(x.tocsc()).tocsr()
            to_stack = []
            if self.include_bias:
                to_stack.append(np.ones(shape=(n_samples, 1), dtype=x.dtype))
            to_stack.append(x)
            for deg in range(2, self.degree + 1):
                xp_next = _csr_polynomial_expansion(
                    x.data,
                    x.indices,
                    x.indptr,
                    x.shape[1],
                    self.interaction_only,
                    deg,
                )
                if xp_next is None:
                    break
                to_stack.append(xp_next)
            xp = sparse.hstack(to_stack, format="csr")
        elif sparse.isspmatrix_csc(x) and self.degree < 4:
            return self.transform(x.tocsr()).tocsc()
        else:
            combinations = self._combinations(
                n_features,
                self.degree,
                self.interaction_only,
                self.include_bias,
            )
            if sparse.isspmatrix(x):
                columns = []
                for comb in combinations:
                    if comb:
                        out_col = 1
                        for col_idx in comb:
                            out_col = x[:, col_idx].multiply(out_col)
                        columns.append(out_col)
                    else:
                        bias = sparse.csc_matrix(np.ones((x.shape[0], 1)))
                        columns.append(bias)
                xp = sparse.hstack(columns, dtype=x.dtype).tocsc()
            else:
                xp = np.empty(
                    (n_samples, self.n_output_features_),
                    dtype=x.dtype,
                    order=self.order,
                )
                for i, comb in enumerate(combinations):
                    xp[:, i] = x[:, comb].prod(1)

        return xp

    def inverse(self, y):
        """
        Invert the transformation.

        This function satisfies
        :code:`self.inverse(self.transform(x)) == x`

        Parameters
        ----------
        y: numpy.ndarray, shape (n_samples, n_output_features)
            Data to which to apply the inverse.
            Must have the same number of features as the transformed data

        Returns
        -------
        x: numpy.ndarray, shape (n_samples, n_input_features)
            Output of inverse map applied to y.
        """
        check_is_fitted(self, "n_output_features_")
        if y.shape[1] != self.n_output_features_:
            raise ValueError(
                "y has the wrong number of features (columns)."
                f"Expected {self.n_output_features_}, received {y.shape[1]}"
            )
        if self.include_bias:
            return y[:, 1 : self.n_input_features_ + 1]
        else:
            return y[:, : self.n_input_features_]

    @staticmethod
    def _combinations(n_features, degree, interaction_only, include_bias):
        comb = combinations if interaction_only else combinations_w_r
        start = int(not include_bias)
        return chain.from_iterable(
            comb(range(n_features), i) for i in range(start, degree + 1)
        )

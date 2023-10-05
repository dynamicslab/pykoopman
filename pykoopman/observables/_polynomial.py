"""moduel for Polynomial observables"""
from __future__ import annotations

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
from ..common import validate_input
from ._base import BaseObservables


class Polynomial(PolynomialFeatures, BaseObservables):
    """
    Polynomial observables.

    This is essentially the `sklearn.preprocessing.PolynomialFeatures` with support for
    complex numbers.

    Args:
        degree (int, optional): The degree of the polynomial features. Default is 2.
        interaction_only (bool, optional): If True, only interaction features are
            produced: features that are products of at most ``degree`` *distinct*
            input features (so not ``x[1] ** 2``, ``x[0] * x[2] ** 3``,
            etc.). Default is False.
        include_bias (bool, optional): If True, then include a bias column, the feature
            in which all polynomial powers are zero (i.e., a column of ones - acts as an
            intercept term in a linear model). Default is True.
        order (str in {'C', 'F'}, optional): Order of output array in the dense case.
            'F' order is faster to compute, but may slow down subsequent estimators.
            Default is 'C'.

    Raises:
        ValueError: If degree is less than 1.
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True, order="C"):
        """
        Initialize the Polynomial object.

        Args:
            degree (int, optional): The degree of the polynomial features. Default is 2.
            interaction_only (bool, optional): If True, only interaction features are
                produced: features that are products of at most ``degree`` *distinct*
                input features (so not ``x[1] ** 2``, ``x[0] * x[2] ** 3``,
                etc.). Default is False.
            include_bias (bool, optional): If True, then include a bias column, the
                feature in which all polynomial powers are zero (i.e., a column of
                ones - acts as an intercept term in a linear model). Default is True.
            order (str in {'C', 'F'}, optional): Order of output array in the dense
                case. 'F' order is faster to compute, but may slow down subsequent
                estimators. Default is 'C'.

        Raises:
            ValueError: If degree is less than 1.
        """
        if degree == 0:
            raise ValueError(
                "degree must be at least 1, otherwise inverse cannot be " "computed"
            )
        super(Polynomial, self).__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order=order,
        )
        self.include_state = True

    def fit(self, x, y=None):
        """
        Compute number of output features.

        This method fits the `Polynomial` instance to the input data `x`. It calls the
        `fit` method of the superclass (`PolynomialFeatures` from
        `sklearn.preprocessing`), which computes the number of output features based
        on the degree of the polynomial and the interaction_only flag. It also sets
        `n_input_features_` and `n_output_features_` attributes. Then, it initializes
        `measurement_matrix_` as a zero matrix of size `n_input_features_` by
        `n_output_features_` and sets the main diagonal to 1, depending on the
        `include_bias` attribute. The input `y` is not used in this method; it is
        only included to maintain compatibility with the usual interface of `fit`
        methods in scikit-learn.

        Args:
            x (np.ndarray): The measurement data to be fit, with shape (n_samples,
                n_features).
            y (array-like, optional): Dummy input. Defaults to None.

        Returns:
            self: A fit instance of `Polynomial`.

        Raises:
            ValueError: If the input data is not valid.
        """
        x = validate_input(x)
        self.n_consumed_samples = 0

        y_poly_out = super(Polynomial, self).fit(x.real, y)

        self.measurement_matrix_ = np.zeros([x.shape[1], y_poly_out.n_output_features_])
        if self.include_bias:
            self.measurement_matrix_[:, 1 : 1 + x.shape[1]] = np.eye(x.shape[1])
        else:
            self.measurement_matrix_[:, : x.shape[1]] = np.eye(x.shape[1])

        return y_poly_out

    def transform(self, x):
        """
        Transforms the data to polynomial features.

        This method transforms the data `x` into polynomial features. It first checks if
        the fit method has been called by checking the `n_input_features_` attribute,
        then it validates the input `x`. If `x` is a CSR sparse matrix and the degree is
        less than 4, it uses a method based on "Leveraging Sparsity to Speed Up
        Polynomial Feature Expansions of CSR Matrices Using K-Simplex Numbers" by
        Andrew Nystrom and John Hughes. If `x` is a CSC sparse matrix and the degree
        is less than 4, it converts `x` to CSR, generates the polynomial features,
        then converts back to CSC. For dense arrays or CSC sparse matrix with a
        degree of 4 or more, it generates the polynomial features through a slower
        process.

        Args:
            x (array-like or CSR/CSC sparse matrix): The data to transform, row by row.
            The shape should be (n_samples, n_features). Prefer CSR over CSC for
            sparse input (for speed), but CSC is required if the degree is 4 or higher.

        Returns:
            xp (np.ndarray or CSR/CSC sparse matrix): The matrix of features, where
            n_output_features is the number of polynomial features generated from the
            combination of inputs. The shape is (n_samples, n_output_features).

        Raises:
            ValueError: If the input data is not valid or the shape of `x` does not
            match training shape.
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

    @staticmethod
    def _combinations(n_features, degree, interaction_only, include_bias):
        """
        Generate combinations for polynomial features.

        This static method generates combinations of features for the polynomial
        transformation. The combinations depend on whether interaction_only is set
        and whether a bias term should be included.

        Args:
            n_features (int): The number of features.
            degree (int): The degree of the polynomial.
            interaction_only (bool): If True, only interaction features are produced.
            include_bias (bool): If True, a bias column is included.

        Returns:
            itertools.chain: An iterable over all combinations.
        """
        comb = combinations if interaction_only else combinations_w_r
        start = int(not include_bias)
        return chain.from_iterable(
            comb(range(n_features), i) for i in range(start, degree + 1)
        )

    @property
    def powers_(self):
        """
        Get the exponent for each of the inputs in the output.

        This property method returns the exponents for each input feature in the
        polynomial output. It first checks whether the model has been fitted, then
        uses the `_combinations` method to get the combinations of features, and
        finally calculates the exponents for each input feature.

        Returns:
            np.ndarray: A 2D array where each row represents a feature and each
            column represents an output of the polynomial transformation. The
            values are the exponents of the input features.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        check_is_fitted(self)

        combinations = self._combinations(
            n_features=self.n_input_features_,
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        return np.vstack(
            [np.bincount(c, minlength=self.n_input_features_) for c in combinations]
        )

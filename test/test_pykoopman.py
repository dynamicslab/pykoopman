from sklearn.utils.validation import check_is_fitted

from pykoopman import Koopman


# Dummy test
def test_pykoopman():
    model = Koopman()
    assert isinstance(model, Koopman)


def test_fit(data_random):
    x = data_random
    model = Koopman()
    model.fit(x)
    check_is_fitted(model)


def test_koopman_matrix(data_random):
    x = data_random
    model = Koopman()
    model.fit(x)
    assert model.koopman_matrix.shape[0] == model.n_output_features_

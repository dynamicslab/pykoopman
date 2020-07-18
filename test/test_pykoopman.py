from pykoopman import Koopman


# Dummy test
def test_pykoopman():
    model = Koopman()
    assert isinstance(model, Koopman)

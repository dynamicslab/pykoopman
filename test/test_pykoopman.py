from pykoopman import PyKoopman


# Dummy test
def test_pykoopman():
    model = PyKoopman()
    assert isinstance(model, PyKoopman)

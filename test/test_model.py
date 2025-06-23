from keras.layers import Input
from src.model import standard_unit, Nest_Net

def test_standard_unit():
    inp = Input(shape=(32, 32, 1))
    out = standard_unit(inp, stage='test', nb_filter=8)
    assert out is not None

def test_nest_net():
    model = Nest_Net(32, 32, 1)
    assert model is not None
    assert hasattr(model, "compile")

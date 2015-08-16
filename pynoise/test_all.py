from pynoise.noisemodule import *

def test_const():
    const = Const(1)

    assert const.get_value(0,3,2) == 1

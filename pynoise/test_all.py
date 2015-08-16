from pynoise.noisemodule import *

def test_const():
    const = Const(1)

    assert const.get_value(0,0,0) == 1

def test_abs():
    const0 = Const(-1)
    const1 = Const(1)

    abs0 = Abs(const0)
    abs1 = Abs(const1)

    assert abs0.get_value(0,0,0) == 1
    assert abs1.get_value(0,0,0) == 1

def test_add():
    const0 = Const(1)
    const1 = Const(2)
    const2 = Const(3)

    add0 = Add(const0, const1)
    add1 = Add(const1, const2)

    assert add0.get_value(0,0,0) == 3
    assert add1.get_value(0,0,0) == 5

def test_billow():
    billow = Billow()

def test_blend():
    const0 = Const(0)
    const1 = Const(1)

    blend = Blend(const0, const1, const0)

    assert blend.get_value(0,0,0) == 0.5

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

def test_checkerboard():
    checkerboard = Checkerboard()

def test_clamp():
    const0 = Const(0.75)

    clamp0 = Clamp(const0, upper_bound=0.5)
    clamp1 = Clamp(const0, lower_bound=0.8)
    clamp2 = Clamp(const0)

    assert clamp0.get_value(0,0,0) == 0.5
    assert clamp1.get_value(0,0,0) == 0.8
    assert clamp2.get_value(0,0,0) == 0.75

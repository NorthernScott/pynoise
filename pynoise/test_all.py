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

def test_curve():
    const0 = Const(1)
    points = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    curve = Curve(const0, points=points)
    curve.get_value(0,0,0)

    curve1 = Curve(const0)
    curve1.add_control_point(0.1, 0.2)
    curve1.add_control_point(0.2, 0.3)
    curve1.add_control_point(0.3, 0.4)
    curve1.add_control_point(0.4, 0.5)
    curve.get_value(0,0,0)

def test_cylinders():
    cylinders = Cylinders()
    cylinders1 = Cylinders(frequency=4)
    assert cylinders1.get_value(0,0,0) == cylinders.get_value(0,0,0)

def test_displace():
    c0 = Const(0)
    b1 = Billow()

    displace = Displace(c0, c0, c0, b1)

    assert b1.get_value(0,0,0) == displace.get_value(0,0,0)

def test_exponent():
    c0 = Const(-1)
    c1 = Const(0.5)
    c2 = Const(1)

    e0 = Exponent(c0, exponent=2)
    e1 = Exponent(c1, exponent=2)
    e2 = Exponent(c2, exponent=2)

    assert e0.get_value(0,0,0) == -1
    assert e1.get_value(0,0,0) == 0.125
    assert e2.get_value(0,0,0) == 1

def test_invert():
    c0 = Const(-1)
    c1 = Const(1)

    i0 = Invert(c0)
    i1 = Invert(c1)

    assert i0.get_value(0,0,0) == 1
    assert i1.get_value(0,0,0) == -1

def test_max():
    c0 = Const(1)
    c1 = Const(2)
    c2 = Const(3)

    m0 = Max(c0, c1)
    m1 = Max(c2, c1)

    assert m0.get_value(0,0,0) == 2
    assert m1.get_value(0,0,0) == 3

def test_min():
    c0 = Const(1)
    c1 = Const(2)
    c2 = Const(3)

    m0 = Min(c0, c1)
    m1 = Min(c2, c1)

    assert m0.get_value(0,0,0) == 1
    assert m1.get_value(0,0,0) == 2

def test_multiply():
    c0 = Const(2)

    m0 = Multiply(c0, c0)
    assert m0.get_value(0,0,0) == 4

def test_perlin():
    perlin = Perlin()
    perlin.get_value(0,0,0)

def test_power():
    c0 = Const(-1)
    c1 = Const(0.5)
    c2 = Const(2)

    p1 = Power(c0, c2)
    p2 = Power(c1, c2)
    p3 = Power(c2, c2)

    assert p1.get_value(0,0,0) == 1
    assert p2.get_value(0,0,0) == 0.25
    assert p3.get_value(0,0,0) == 4

def test_ridged():
    r0 = RidgedMulti()
    r0.get_value(0,0,0)

def test_rotate():
    c0 = Const(1)

    r0 = RotatePoint(c0)
    r0.get_value(0,0,0)

def test_scalebias():
    c0 = Const(2)

    s0 = ScaleBias(c0, scale=2, bias=0.5)
    s1 = ScaleBias(c0, bias=3)

    assert s0.get_value(0,0,0) == 4.5
    assert s1.get_value(0,0,0) == 5

def test_scalepoint():
    c0 = Const(2)

    s0 = ScalePoint(c0)
    s0.get_value(0,0,0)

def test_select():
    c0 = Const(0)
    c1 = Const(1)
    c3 = Const(0.5)
    c4 = Const(-0.5)

    s0 = Select(c0, c1, c3)
    s1 = Select(c0, c1, c4, lower_bound=0)
    assert s0.get_value(0,0,0) == 1
    assert s1.get_value(0,0,0) == 0

def test_spheres():
    s0 = Spheres()
    s1 = Spheres(frequency=10)
    assert s0.get_value(0,0,0) == s1.get_value(0,0,0)

def test_terrace():
    c0 = Const(0)

    t0 = Terrace(c0, control_points=[-1,-0.5, 0, 0.5, 1])
    t0.get_value(0,0,0)

def test_translatepoint():
    c0 = Const(.1)
    tp = TranslatePoint(c0)
    tp.get_value(0,0,0)

def test_turbulence():
    c0 = Const(1)
    t0 = Turbulence(c0)
    t0.get_value(0,0,0)

def test_voronoi():
    v = Voronoi()
    v.get_value(0,0,0)

def test_util_clamp():
    assert clamp(1,0,2) == 1
    assert clamp(0,1,2) == 1
    assert clamp(2,0,1) == 1

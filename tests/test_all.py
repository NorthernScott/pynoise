import sys
sys.path.insert(1, '../pynoise/')

from pynoise.noisemodule import *
from pynoise.noiseutil import *
import pytest

def within(a, b, epsilon=0.000001):
    return abs(a - b) < epsilon

def check_self(mod):
    assert within(mod.get_value(0,0,0), mod.get_values(1,1, 0,0, 0,0, 0,0))

def test_noise_module():
    n = NoiseModule()

    with pytest.raises(NotImplementedError):
        n.get_value(0,0,0)

def test_const():
    const = Const(1)

    assert const.get_value(0,0,0) == 1
    assert within(const.get_value(0,0,0), const.get_values(1,1, 0,0, 0,0, 0,0))

def test_abs():
    const0 = Const(-1)
    const1 = Const(1)

    abs0 = Abs(const0)
    abs1 = Abs(const1)

    assert abs0.get_value(0,0,0) == 1
    assert abs1.get_value(0,0,0) == 1

    check_self(abs0)
    check_self(abs1)

def test_add():
    const0 = Const(1)
    const1 = Const(2)
    const2 = Const(3)

    add0 = Add(const0, const1)
    add1 = Add(const1, const2)

    p0 = Perlin()
    p1 = Perlin(seed=2)

    add2 = Add(p0, p1)

    assert add0.get_value(0,0,0) == 3
    assert add1.get_value(0,0,0) == 5

    check_self(add0)
    check_self(add1)

def test_billow():
    billow = Billow()
    check_self(billow)

def test_blend():
    const0 = Const(0)
    const1 = Const(1)

    p0 = Perlin()
    p1 = Perlin(seed=3)

    blend = Blend(const0, const1, const0)
    b1 = Blend(p0, p1, p0)

    assert blend.get_value(0,0,0) == 0.5

    check_self(b1)
    check_self(blend)

def test_checkerboard():
    checkerboard = Checkerboard()
    checkerboard.get_value(0,0,0)
    checkerboard.get_value(1.51,1.51,1.51)
    check_self(checkerboard)

def test_clamp():
    const0 = Const(0.75)

    clamp0 = Clamp(const0, upper_bound=0.5)
    clamp1 = Clamp(const0, lower_bound=0.8)
    clamp2 = Clamp(const0)

    assert clamp0.get_value(0,0,0) == 0.5
    assert clamp1.get_value(0,0,0) == 0.8
    assert clamp2.get_value(0,0,0) == 0.75
    check_self(clamp0)
    check_self(clamp1)
    check_self(clamp2)

def test_curve():
    const0 = Const(1)
    points = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    curve = Curve(const0, points=points)
    check_self(curve)

    curve1 = Curve(const0)
    curve1.add_control_point(0.1, 0.2)
    curve1.add_control_point(0.2, 0.3)
    curve1.add_control_point(0.3, 0.4)
    curve1.add_control_point(0.4, 0.5)
    check_self(curve1)

    curve1.clear_control_points()
    curve1.add_control_point(0.1, 0.2)
    curve1.add_control_point(0.2, 0.3)
    curve1.add_control_point(0.3, 0.4)
    curve1.add_control_point(0.4, 0.5)
    check_self(curve1)

    const1 = Const(0)
    curve2 = Curve(const1)
    curve2.add_control_point(0.1, 0.2)
    curve2.add_control_point(0.2, 0.3)
    curve2.add_control_point(0.3, 0.4)
    curve2.add_control_point(0.4, 0.5)
    check_self(curve2)

def test_cylinders():
    cylinders = Cylinders()
    cylinders1 = Cylinders(frequency=4)
    assert cylinders1.get_value(0,0,0) == cylinders.get_value(0,0,0)
    check_self(cylinders)
    check_self(cylinders1)

def test_displace():
    c0 = Const(0)
    b1 = Billow()

    displace = Displace(c0, c0, c0, b1)

    assert b1.get_value(0,0,0) == displace.get_value(0,0,0)
    check_self(displace)

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

    check_self(e0)
    check_self(e1)
    check_self(e2)

def test_invert():
    c0 = Const(-1)
    c1 = Const(1)

    i0 = Invert(c0)
    i1 = Invert(c1)

    assert i0.get_value(0,0,0) == 1
    assert i1.get_value(0,0,0) == -1

    check_self(i0)
    check_self(i1)

def test_max():
    c0 = Const(1)
    c1 = Const(2)
    c2 = Const(3)

    m0 = Max(c0, c1)
    m1 = Max(c2, c1)

    assert m0.get_value(0,0,0) == 2
    assert m1.get_value(0,0,0) == 3

    check_self(m0)
    check_self(m1)

def test_min():
    c0 = Const(1)
    c1 = Const(2)
    c2 = Const(3)

    m0 = Min(c0, c1)
    m1 = Min(c2, c1)

    assert m0.get_value(0,0,0) == 1
    assert m1.get_value(0,0,0) == 2

    check_self(m0)
    check_self(m1)

def test_multiply():
    c0 = Const(2)

    m0 = Multiply(c0, c0)
    assert m0.get_value(0,0,0) == 4

    check_self(m0)

def test_perlin():
    perlin = Perlin()
    perlin.get_value(0,0,0)

    check_self(perlin)

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

    check_self(p3)
    check_self(p1)
    check_self(p2)

def test_ridged():
    r0 = RidgedMulti()
    r0.get_value(0,0,0)

    r1 = RidgedMulti(gain=100, octaves=30)
    r1.get_value(0,0,0)

    r2 = RidgedMulti(gain=-100, octaves=30)
    r2.get_value(0,0,0)

    check_self(r0)
    check_self(r1)
    check_self(r2)

def test_rotate():
    c0 = Const(1)

    r0 = RotatePoint(c0)
    r0.get_value(0,0,0)

    check_self(r0)

def test_scalebias():
    c0 = Const(2)

    s0 = ScaleBias(c0, scale=2, bias=0.5)
    s1 = ScaleBias(c0, bias=3)

    assert s0.get_value(0,0,0) == 4.5
    assert s1.get_value(0,0,0) == 5

    check_self(s0)
    check_self(s1)

def test_scalepoint():
    c0 = Const(2)

    s0 = ScalePoint(c0)
    s0.get_value(0,0,0)

    check_self(s0)

def test_select():
    c0 = Const(0)
    c1 = Const(1)
    c3 = Const(0.5)
    c4 = Const(-0.5)

    s0 = Select(c0, c1, c3)
    s1 = Select(c0, c1, c4, lower_bound=0)
    assert s0.get_value(0,0,0) == 1
    assert s1.get_value(0,0,0) == 0

    s2 = Select(c0,c1,c3, edge_falloff=0.5)
    s2.get_value(0,0,0)

    s3 = Select(c0,c1,c4, edge_falloff=0.5)
    s3.get_value(0,0,0)

    s4 = Select(c0,c1,c4, edge_falloff=0.5, lower_bound=0.9)
    s4.get_value(0,0,0)

    s5 = Select(c0,c1,c3, edge_falloff=0.5, lower_bound=0.9)
    s5.get_value(0,0,0)

    c5 = Const(10)
    s6 = Select(c0,c1,c5, edge_falloff=0.5)
    s6.get_value(0,0,0)

    check_self(s0)
    check_self(s1)
    check_self(s2)
    check_self(s3)
    check_self(s4)
    check_self(s5)

def test_spheres():
    s0 = Spheres()
    s1 = Spheres(frequency=10)
    assert s0.get_value(0,0,0) == s1.get_value(0,0,0)

    check_self(s0)
    check_self(s1)

def test_terrace():
    c0 = Const(0)

    t0 = Terrace(c0, control_points=[-1,-0.5, 0, 0.5, 1])
    t0.get_value(0,0,0)

    t1 = Terrace(c0, control_points=[1, 2])
    t1.get_value(0,0,0)

    t2 = Terrace(c0, control_points=[-1,-0.5, 0, 0.5, 1], invert_terraces=True)
    t2.get_value(0,0,0)

    check_self(t0)
    check_self(t1)
    check_self(t2)

def test_translatepoint():
    c0 = Const(.1)
    tp = TranslatePoint(c0)
    tp.get_value(0,0,0)

    check_self(tp)

def test_turbulence():
    c0 = Const(1)
    t0 = Turbulence(c0)

    p0 = Perlin()
    t1 = Turbulence(p0)

    t0.get_value(0,0,0)
    check_self(t1)

def test_voronoi():
    v = Voronoi()
    v.get_value(0,0,0)

    v1 = Voronoi(enable_distance=True)
    v1.get_value(0,0,0)

    check_self(v)
    check_self(v1)

def test_util_clamp():
    assert clamp(1,0,2) == 1
    assert clamp(0,1,2) == 1
    assert clamp(2,0,1) == 1

def test_linear_interp():
    assert abs(linear_interp(0, 1, 0.5) - 0.5) <= 0.000001
    assert abs(linear_interp(0, 1, 0.1) - 0.1) <= 0.000001
    assert abs(linear_interp(0, 1, 0.25) - 0.25) <= 0.000001
    assert abs(linear_interp(0, 1, 0.63) - 0.63) <= 0.000001

    assert abs(linear_interp(0.1, 1, 0.31) - 0.379) <= 0.000001
    assert abs(linear_interp(0.2, 1, 0.46) - 0.568) <= 0.000001
    assert abs(linear_interp(0.3, 1, 0.19) - 0.433) <= 0.000001
    assert abs(linear_interp(0.4, 1, 0.29) - 0.574) <= 0.000001
    assert abs(linear_interp(0.5, 1, 0.15) - 0.575) <= 0.000001

def test_noise_map_sphere_gpu():
    p = Perlin()
    t = Turbulence(p)
    noise_map_sphere_gpu(east_bound=180, west_bound=-180,
    north_bound=1200, south_bound=-1200, width=512, height=256, source=t)

import sys
sys.path.insert(1, '../pynoise/')

from pynoise.noisemodule import *
from pynoise.noiseutil import *
import pytest
import PIL

width, height = 32, 32

def compare_images(path1, path2):
    i1 = Image.open(path1)
    i2 = Image.open(path2)

    pairs = zip(i1.getdata(), i2.getdata())
    dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))
    ncomponents = i1.size[0] * i1.size[1] * 3
    percent = ((dif / 255.0 * 100) / ncomponents)
    assert percent < 5

def test_plane_render():
    p = Perlin()
    g = grayscale_gradient()
    nm = noise_map_plane(width=width, height=height, lower_x=0, upper_x=1, lower_z=0, upper_z=1, source=p)

    nm2 = noise_map_plane_gpu(width=width, height=height, lower_x=0, upper_x=1, lower_z=0, upper_z=1, source=p)
    render = RenderImage()

    render.render(width, height, nm, 'plane.png', g)
    render.render(width, height, nm2, 'plane_gpu.png', g)

    compare_images('plane.png', 'plane_gpu.png')

def test_sphere_render():
    p = Perlin()
    g = grayscale_gradient()
    nm = noise_map_sphere(width=width*2, height=height, east_bound=180, west_bound=-180, north_bound=90, south_bound=-90, source=p)
    nm2 = noise_map_sphere_gpu(width=width*2, height=height, east_bound=180, west_bound=-180, north_bound=90, south_bound=-90, source=p)

    # smap(width=width*2, height=height, east_bound=180, west_bound=-180, north_bound=90, south_bound=-90, source=p)
    render = RenderImage()
    render.render(width*2, height, nm, 'sphere.png', g)
    render.render(width*2, height, nm2, 'sphere_gpu.png', g)

    compare_images('sphere.png', 'sphere_gpu.png')

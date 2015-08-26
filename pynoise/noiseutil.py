from sortedcontainers import SortedDict
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import numpy as np
from PIL import Image, ImageColor
from pynoise.util import clamp
from pynoise.interpolators import linear_interp
import math

def linear_interp_color(color0, color1, alpha):
    c0_lab = convert_color(color0, LabColor)
    c1_lab = convert_color(color1, LabColor)

    nc_l = linear_interp(c0_lab.lab_l, c1_lab.lab_l, alpha)
    nc_a = linear_interp(c0_lab.lab_a, c1_lab.lab_a, alpha)
    nc_b = linear_interp(c0_lab.lab_b, c1_lab.lab_b, alpha)

    return convert_color(LabColor(nc_l, nc_a, nc_b), sRGBColor)

class GradientColor():
    gradient_points = SortedDict()

    def add_gradient_point(self, position, color):
        self.gradient_points[position] = color

    def get_color(self, position):
        assert (len(self.gradient_points.keys()) > 1)

        for i, p in enumerate(self.gradient_points.keys()):
            if position < p:
                break

        index0 = clamp(i - 1, 0, len(self.gradient_points)-1)
        index1 = clamp(i, 0, len(self.gradient_points)-1)

        if index0 == index1:
            return self.gradient_points[self.gradient_points.keys()[index1]]

        input0 = self.gradient_points.keys()[index0]
        input1 = self.gradient_points.keys()[index1]

        alpha = (position - input0) / (input1 - input0)
        c0 = self.gradient_points[input0]
        c1 = self.gradient_points[input1]

        return linear_interp_color(c0, c1, alpha)

def noise_map_cylinder(lower_angle=0, upper_angle=0, lower_height=0, upper_height=0,
    width=0, height=0, source=None):
    assert lower_angle < upper_angle
    assert lower_height < upper_height
    assert width > 0
    assert height > 0
    assert source is not None
    assert img_name is not None

    angle_extent = upper_angle - lower_angle
    height_extent = upper_height - lower_height
    x_delta = angle_extent / width
    y_delta = angle_extent / height
    cur_angle = lower_angle
    cur_height = lower_height
    nm = np.zeros((height, width))

    for y in range(0, height):
        cur_angle = lower_angle
        for x in range(0, width):
            cx = math.cos(math.radians(cur_angle))
            cy = height
            cz = math.sin(math.radians(cur_angle))
            nm[y][x] = source.get_value(cx, cy, cz)
            cur_angle += x_delta
        cur_height += y_delta

    return np.flipud(nm)

def noise_map_plane(seamless=False, lower_x=0, upper_x=0, lower_z=0, upper_z=0,
    width=0, height=0, source=None):
    assert lower_x < upper_x
    assert lower_z < upper_z
    assert width > 0
    assert height > 0
    assert source is not None

    x_extent = upper_x - lower_x
    z_extent = upper_z - lower_z
    x_delta = x_extent / width
    z_delta = z_extent / height
    x_cur = lower_x
    z_cur = lower_z

    nm = np.zeros((height, width))

    for z in range(width):
        x_cur = lower_x
        for x in range(height):
            if not seamless:
                nm[z][x] = source.get_value(x_cur, 0, z_cur)
            else:
                sw = source.get_value(xCur, 0, zCur)
                se = source.get_value(xCur+x_extent, 0, zCur)
                nw = source.get_value(xCur, 0, zCur+z_extent)
                ne = source.get_value(xCur+x_extent, 0, z_extent)

                x_blend = 1 - ((xCur - lower_x) / x_extent)
                z_blend = 1 - ((zCur - lower_z) / z_extent)

                z0 = linear_interp(sw, se, x_blend)
                z1 = linear_interp(nw, ne, x_blend)
                nm[z][x] = linear_interp(z0, z1, z_blend)
            x_cur += x_delta
        z_cur += z_delta

    return np.flipud(nm)

def noise_map_sphere(east_bound=0, west_bound=0, north_bound=0, south_bound=0,
    width=0, height=0, source=None):
    assert east_bound > west_bound
    assert north_bound > south_bound
    assert width > 0
    assert height > 0
    assert source is not None

    nm = np.zeros((height, width))

    lon_extent = east_bound - west_bound
    lat_extent = north_bound - south_bound
    x_delta = lon_extent / width
    y_delta = lat_extent / height
    cur_lon = west_bound
    cur_lat = south_bound

    for y in range(height):
        cur_lon = west_bound
        for x in range(width):
            r = math.cos(math.radians(cur_lat))
            x = r * math.cos(math.radians(cur_lon))
            y = math.sin(math.radians(cur_lat))
            z = r * math.sin(math.radians(cur_lon))

            nm[y][x] = source.get_value(x, y, z)

            cur_lon += x_delta
        cur_lat += y_delta

    return np.flipud(nm)

def grayscale_gradient():
    grad = GradientColor()
    grad.add_gradient_point(-1, sRGBColor(0, 0, 0, is_upscaled=True))
    grad.add_gradient_point(1, sRGBColor(255, 255, 255, is_upscaled=True))
    return grad

def terrain_gradient():
    grad = GradientColor()

    grad.add_gradient_point(-1.00, sRGBColor(  0,   0, 128, is_upscaled=True))
    grad.add_gradient_point(-0.20, sRGBColor( 32,  64, 128, is_upscaled=True))
    grad.add_gradient_point(-0.04, sRGBColor( 64,  96, 192, is_upscaled=True))
    grad.add_gradient_point(-0.02, sRGBColor(192, 192, 128, is_upscaled=True))
    grad.add_gradient_point( 0.00, sRGBColor(  0, 192,   0, is_upscaled=True))
    grad.add_gradient_point( 0.25, sRGBColor(192, 192,   0, is_upscaled=True))
    grad.add_gradient_point( 0.50, sRGBColor(160,  96,  64, is_upscaled=True))
    grad.add_gradient_point( 0.75, sRGBColor(128, 255, 255, is_upscaled=True))
    grad.add_gradient_point( 1.00, sRGBColor(255, 255, 255, is_upscaled=True))

    return grad

class RenderImage():
    def __init__(self, light_enabled=False, wrap_enabled=False, light_azimuth=45,
        light_brightness=1, light_color=sRGBColor(1, 1, 1),
        light_contrast=1, light_elev=45, light_intensity=1):
        self.light_enabled = light_enabled
        self.wrap_enabled = wrap_enabled
        self.light_azimuth = light_azimuth
        self.light_brightness = light_brightness
        self.light_color = light_color
        self.light_contrast = light_contrast
        self.light_elev = light_elev
        self.light_intensity = light_intensity

        self.cos_azimuth = 0
        self.sin_azimuth = 0
        self.cos_elev = 0
        self.sin_elev = 0
        self.recalc_light = True

    def calc_dest_color(self, source, background, light_value):
        nc = linear_interp_color(background, source, 1)

        if self.light_enabled:
            ncr, ncg, ncb = nc.get_value_tuple()
            lcr, lcg, lcb = self.light_color.get_value_tuple()
            lcr *= light_value
            lcg *= light_value
            lcb *= light_value

            ncr *= lcr
            ncg *= lcg
            ncb *= lcb

            return sRGBColor(ncr, ncg, ncb)
        else:
            return nc

    def calc_light_intensity(self, center, left, right, up, down):
        if self.recalc_light:
            self.cos_azimuth = math.cos(math.radians(self.light_azimuth))
            self.sin_azimuth = math.sin(math.radians(self.light_azimuth))
            self.cos_elev = math.cos(math.radians(self.light_elev))
            self.sin_elev = math.sin(math.radians(self.light_elev))
            self.recalc_light = False

        I_MAX = 1
        io = I_MAX * math.sqrt(2) * self.sin_elev / 2
        ix = (I_MAX - io) * self.light_contrast * math.sqrt(2) * self.cos_elev * self.cos_azimuth
        iy = (I_MAX - io) * self.light_contrast * math.sqrt(2) * self.cos_elev * self.sin_azimuth
        intensity = (ix * (left - right) + iy * (down - up) + io)

        if intensity < 0:
            return 0
        else:
            return intensity

    def render(self, noisemap, image_name, gradient):
        height, width = noisemap.shape
        img = Image.new('RGB', (width, height), '#ffffff')

        for y in range(height):
            for x in range(width):
                dest_color = gradient.get_color(noisemap[y][x])

                light_intensity = 1

                if self.light_enabled:
                    x_left = 0
                    x_right = 0
                    y_up = 0
                    y_down = 0

                    if(self.wrap_enabled):
                        if x == 0:
                            x_left = width-1
                            x_right = 1
                        elif x == width-1:
                            x_left = -1
                            x_right = width-1
                        else:
                            x_left = -1
                            x_right = 1

                        if y == 0:
                            y_down = height-1
                            y_up = 1
                        elif y == height-1:
                            y_down = -1
                            y_up = -(height-1)
                        else:
                            y_down = -1
                            y_up = 1
                    else:
                        if x == 0:
                            x_left = 0
                            x_right = 1
                        elif x == width - 1:
                            x_left = -1
                            x_right = 0
                        else:
                            x_left = -1
                            x_right = 1

                        if y == 0:
                            y_down = 0
                            y_up = 1
                        elif y == height - 1:
                            y_down = -1
                            y_up = 0
                        else:
                            y_down = -1
                            y_up = 1

                    nc = noisemap[y][x]
                    nl = noisemap[y][x+x_left]
                    nr = noisemap[y][x+x_right]
                    nd = noisemap[y+y_down][x]
                    nu = noisemap[y+y_up][x]

                    light_intensity = self.calc_light_intensity(nc, nl, nr, nu, nd)
                    light_intensity *= self.light_brightness

                bg_color = sRGBColor(1, 1, 1)
                color = self.calc_dest_color(dest_color, bg_color, light_intensity)
                t = color.get_upscaled_value_tuple()
                img.putpixel((x,y), (t[0], t[1], t[2]))
        img.save(image_name, 'PNG')

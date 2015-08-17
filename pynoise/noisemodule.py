import math

from pynoise.quality import Quality
from pynoise.noise import gradient_coherent_noise_3d
from pynoise.interpolators import linear_interp, cubic_interp
from pynoise.util import clamp

from sortedcontainers import SortedDict, SortedList


class NoiseModule():
    def get_value(self, x, y, z):
        raise NotImplementedError('Not Implemented.')

class Abs(NoiseModule):
    """ Returns the absolute value of the given source module. """
    def __init__(self, source0):
        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        return abs(self.source0.get_value(x, y, z))

class Add(NoiseModule):
    """ Adds the two given source modules together. """
    def __init__(self, source0, source1):
        self.source0 = source0
        self.source1 = source1

    def get_value(self, x, y, z):
        assert (self.source0 is not None)
        assert (self.source1 is not None)

        return self.source0.get_value(x, y, z) + self.source1.get_value(x, y, z)

class Billow(NoiseModule):
    """ Noise function that is more suitable for items like clouds or rocks.
    This module is nearly identical to Perlin noise, however each octave has
    the absolute value taken from the signal.
    """
    def __init__(self, frequency=1, lacunarity=2, quality=Quality.std,
        octaves=6, persistence=0.5, seed=0):

        self.frequency = frequency
        self.lacunarity = lacunarity
        self.quality = quality
        self.octaves = octaves
        self.persistence = persistence
        self.seed = seed

    def get_value(self, x, y, z):
        value = 0
        signal = 0
        curPersistence = 1.0
        seed = 0
        for octave in range(self.octaves):
            seed = self.seed + octave
            signal = gradient_coherent_noise_3d(x, y, z, seed, self.quality)
            signal = 2 * abs(signal) - 1.0
            value += signal * curPersistence

            x *= self.lacunarity
            y *= self.lacunarity
            z *= self.lacunarity

            curPersistence *= self.persistence

        value += 0.5

        return value

class Blend(NoiseModule):
    """ Blends source0 and source1 by performing a linear interpolation with
    source2 acting as the alpha
    """
    def __init__(self, source0, source1, source2):
        self.source0 = source0
        self.source1 = source1
        self.source2 = source2

    def get_value(self, x, y, z):
        assert (self.source0 is not None)
        assert (self.source1 is not None)
        assert (self.source2 is not None)

        v0 = self.source0.get_value(x, y, z)
        v1 = self.source1.get_value(x, y, z)
        alpha = (self.source2.get_value(x, y, z) + 1) / 2

        return linear_interp(v0, v1, alpha)

class Checkerboard(NoiseModule):
    def get_value(self, x, y, z):
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        iz = int(math.floor(z))

        if ((ix & 1 ^ iy & 1 ^ iz & 1) != 0):
            return -1
        else:
            return 1

class Clamp():
    """ Clamps a source between lower_bound and upper_bound.
    lower_bound is defaulted to -1 and upper_bound defaults to 1.
    """
    def __init__(self, source0, lower_bound=-1, upper_bound=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        value = self.source0.get_value(x, y, z)

        if value < self.lower_bound:
            return self.lower_bound
        elif value > self.upper_bound:
            return self.upper_bound
        else:
            return value

class Const(NoiseModule):
    def __init__(self, const):
        self.const = const

    def get_value(self, x, y, z):
        return self.const

class Curve(NoiseModule):
    """ Maps the output of source0 to a cubic spline.
    points is a list of tuples that specify input and outputs.
    Example: [(0, 0.1), (0.1,0.2),...]

    A minimum of 4 points must be added.
    """
    def __init__(self, source0, points=None):
        self.control_points = SortedDict()
        self.source0 = source0

        if points is not None:
            for point in points:
                self.add_control_point(point[0], point[1])

    def add_control_point(self, input, output):
        self.control_points[input] = output

    def clear_control_points(self):
        self.control_points = SortedDict()

    def get_value(self, x, y, z):
        assert(self.source0 is not None)
        assert(len(self.control_points.keys()) >= 4)

        value = self.source0.get_value(x, y, z)

        for i, k in enumerate(self.control_points.keys()):
            if value < k:
                break

        index0 = clamp(i - 2, 0, len(self.control_points.keys())-1)
        index1 = clamp(i - 1, 0, len(self.control_points.keys())-1)
        index2 = clamp(i,     0, len(self.control_points.keys())-1)
        index3 = clamp(i + 1, 0, len(self.control_points.keys())-1)

        if index1 == index2:
            return self.control_points[k]

        l = list(self.control_points.keys())

        input0 = l[index1]
        input1 = l[index2]
        alpha = (value - input0) / (input1 - input0)

        return cubic_interp(
            self.control_points[l[index0]],
            self.control_points[l[index1]],
            self.control_points[l[index2]],
            self.control_points[l[index3]],
            alpha
        )

class Cylinders(NoiseModule):
    """ Outputs a series of concentric cyliners centered around 0,y,0.
    There is a new cylinder every `frequency` lengths from origin. Default
    frequency is 1.
    """
    def __init__(self, frequency=1):
        self.frequency = frequency

    def get_value(self, x, y, z):
        x *= self.frequency
        z *= self.frequency

        center = (x**2 + z**2)**0.5
        small = center - math.floor(center)
        large = 1 - small
        nearest = min(small, large)

        return 1.0 - (nearest * 4.0)

class Displace(NoiseModule):
    """ Displaces source3 output by the outputs of source0, source1, and source2.
    source0 adjusts the x values. source1 adjusts the y value and source2 adjusts
    the z value.
    """
    def __init__(self, source0, source1, source2, source3):
        self.source0 = source0
        self.source1 = source1
        self.source2 = source2
        self.source3 = source3

    def get_value(self, x, y, z):
        assert(self.source0 is not None)
        assert(self.source1 is not None)
        assert(self.source2 is not None)
        assert(self.source3 is not None)

        xD = x + (self.source0.get_value(x, y, z))
        yD = y + (self.source1.get_value(x, y, z))
        zD = z + (self.source2.get_value(x, y, z))

        return self.source3.get_value(xD, yD, zD)

class Exponent(NoiseModule):
    """  Sets the exponent value to apply to the output value from the
    source module.

    Because most noise modules will output values that range from -1.0
    to +1.0, this noise module first normalizes this output value (the
    range becomes 0.0 to 1.0), maps that value onto an exponential
    curve, then rescales that value back to the original range.
    """
    def __init__(self, source0, exponent=1):
        self.exponent = exponent
        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        value = self.source0.get_value(x, y, z)

        return math.pow(abs((value + 1) / 2), self.exponent) * 2 - 1

class Invert(NoiseModule):
    """ Inverts the sign of the given source. """
    def __init__(self, source0):
        self.source0 = source0

    def get_value(self, x, y, z):
        assert(self.source0 is not None)

        return -(self.source0.get_value(x, y, z))

class Max(NoiseModule):
    """ Chooses the large value of source0 and source1. """
    def __init__(self, source0, source1):
        self.source0 = source0
        self.source1 = source1

    def get_value(self, x, y, z):
        assert(self.source0 is not None)
        assert(self.source1 is not None)

        v0 = self.source0.get_value(x, y, z)
        v1 = self.source1.get_value(x, y, z)

        return max(v0, v1)

class Min(NoiseModule):
    """ Chooses the lesser value of source0 and source1. """
    def __init__(self, source0, source1):
        self.source0 = source0
        self.source1 = source1

    def get_value(self, x, y, z):
        assert(self.source0 is not None)
        assert(self.source1 is not None)

        v0 = self.source0.get_value(x, y, z)
        v1 = self.source1.get_value(x, y, z)

        return min(v0, v1)

class Multiply(NoiseModule):
    """ Multiplies source0 by source1. """
    def __init__(self, source0, source1):
        self.source0 = source0
        self.source1 = source1

    def get_value(self, x, y, z):
        assert(self.source0 is not None)
        assert(self.source1 is not None)

        return self.source0.get_value(x, y, z) * self.source1.get_value(x, y, z)

class Perlin(NoiseModule):
    """ The classic noise. https://en.wikipedia.org/wiki/Perlin_noise """
    def __init__(self, frequency=1, lacunarity=2, octave=6, persistence=0.5, seed=0, quality=Quality.std):
        self.frequency = frequency
        self.lacunarity = lacunarity
        self.octaves = octave
        self.seed = seed
        self.persistence = persistence
        self.quality = quality

    def get_value(self, x, y, z):
        value = 0.0
        signal = 0.0
        curPersistence = 1.0

        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        for i in range(0, self.octaves):
            seed = (self.seed + i) & 0xffffffff
            signal = gradient_coherent_noise_3d(x, y, z, seed, self.quality)
            value += signal * curPersistence

            x *= self.lacunarity
            y *= self.lacunarity
            z *= self.lacunarity
            curPersistence *= self.persistence

        return value

class Power(NoiseModule):
    """ source0 ** source1. Does not apply scaling. """
    def __init__(self, source0, source1):
        self.source0 = source0
        self.source1 = source1

    def get_value(self, x, y, z):
        assert (self.source0 is not None)
        assert (self.source1 is not None)

        return self.source0.get_value(x,y,z)**self.source1.get_value(x,y,z)

class RidgedMulti(NoiseModule):
    """ This is much like perlin noise, however each octave is modified by
    abs(x*-exponent) where x is x *= frequency repeated over each octave. """
    def __init__(self, frequency=1, lacunarity=2, quality=Quality.std,
        octaves=6, seed=0, exponent=1, offset=1, gain=2):
        self.frequency = frequency
        self.lacunarity = lacunarity
        self.quality = quality
        self.octaves = octaves
        self.seed = seed
        self.exponent = exponent
        self.max_octaves = 30
        self.weights = [0] * self.max_octaves
        self.offset = offset
        self.gain = gain

        freq = 1
        for i in range(0, self.max_octaves):
            self.weights[i] = freq**-exponent
            freq *= self.lacunarity

    def get_value(self, x, y, z):
        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        signal = 0.0
        value = 0.0
        weight = 1.0

        for i in range(self.octaves):
            seed = (self.seed + i) * 0x7fffffff
            signal = gradient_coherent_noise_3d(x, y, z, seed, self.quality)

            signal = abs(signal)
            signal = self.offset = signal

            signal *= signal

            signal *= weight

            weight = signal * self.gain

            if weight > 1:
                weight = 1
            elif weight < 0:
                weight = 0

            value += (signal * self.weights[i])

            x *= self.lacunarity
            y *= self.lacunarity
            z *= self.lacunarity

        return (value * 1.25) - 1

class RotatePoint(NoiseModule):
    """ Rotates source0 around the origin before returning a value. This is a
    right hand system, xAngle increases to the right, yAngle increases upwards
    and zAngle increases inward.
    """
    def __init__(self, source0, xAngle=0, yAngle=0, zAngle=0):
        xCos = math.cos(math.radians(xAngle))
        yCos = math.cos(math.radians(yAngle))
        zCos = math.cos(math.radians(zAngle))

        xSin = math.sin(math.radians(xAngle))
        ySin = math.sin(math.radians(yAngle))
        zSin = math.sin(math.radians(zAngle))

        self.x1a = ySin * xSin * zSin + yCos * zCos
        self.y1a = xCos * zSin
        self.z1a = ySin * zCos - yCos * ySin * zSin

        self.x2a = ySin * xSin * zCos - yCos * zSin
        self.y2a = xCos * zCos
        self.z2a = -yCos * xSin * zCos - ySin * zSin

        self.x3a = -ySin * xCos
        self.y3a = xSin
        self.z3a = yCos * xCos

        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        nx = (self.x1a * x) + (self.y1a * y) + (self.z1a * z)
        ny = (self.x2a * x) + (self.y2a * y) + (self.z2a * z)
        nz = (self.x3a * x) + (self.y3a * y) + (self.z3a * z)

        return self.source0.get_value(nx, ny, nz)

class ScaleBias(NoiseModule):
    """ Takes the value of source0 and multiplies by scale and adds by bias. """
    def __init__(self, source0, bias=0, scale=1):
        self.bias = bias
        self.scale = scale

        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        return self.source0.get_value(x, y, z) * self.scale + self.bias

class ScalePoint(NoiseModule):
    """ Scales the x,y,z before returning source0 value. """
    def __init__(self, source0, sx=1, sy=1, sz=1):
        self.sx = sx
        self.sy = sy
        self.sz = sz

        self.source0 = source0

    def get_value(self, x, y, z):
        assert (self.source0 is not None)

        return self.source0.get_value(x * self.sx, y * self.sy, z * self.sz)

class Select(NoiseModule):
    """
    /// Noise module that outputs the value selected from one of two source
    /// modules chosen by the output value from a control module.
    ///
    /// Unlike most other noise modules, the index value assigned to a source
    /// module determines its role in the selection operation:
    /// - Source module 0 (upper left in the diagram) outputs a value.
    /// - Source module 1 (lower left in the diagram) outputs a value.
    /// - Source module 2 (bottom of the diagram) is known as the <i>control
    ///   module</i>.  The control module determines the value to select.  If
    ///   the output value from the control module is within a range of values
    ///   known as the <i>selection range</i>, this noise module outputs the
    ///   value from the source module with an index value of 1.  Otherwise,
    ///   this noise module outputs the value from the source module with an
    ///   index value of 0.
    ///
    ///
    /// By default, there is an abrupt transition between the output values
    /// from the two source modules at the selection-range boundary.  To
    /// smooth the transition, pass a non-zero value to edge_falloff
    /// method.  Higher values result in a smoother transition.
    ///
    /// This noise module requires three source modules.
    """
    def __init__(self, source0, source1, source2, edge_falloff=0, lower_bound=-1, upper_bound=1):
        self.edge_falloff = edge_falloff
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.source0 = source0
        self.source1 = source1
        self.source2 = source2

    def get_value(self, x, y, z):
        assert (self.source0 is not None)
        assert (self.source1 is not None)
        assert (self.source2 is not None)

        control_value = self.source2.get_value(x, y, z)

        if self.edge_falloff > 0:
            if control_value < (self.lower_bound - self.edge_falloff):
                return s.source0.get_value(x,y,z)

            elif control_value < (self.lower_bound + self.edge_falloff):
                lower_curve = (self.lower_bound - self.edge_falloff)
                upper_curve = (self.lower_bound + self.edge_falloff)
                alpha = scurve3((control_value - lower_curve) / upper_curve - lower_curve)

                return linear_interp(self.source0.get_value(x, y, z),
                    self.source1.get_value(x, y, z), alpha)

            elif control_value < (self.upper_bound - self.edge_falloff):
                return s.source1.get_value(x, y, z)

            elif control_value < (self.upper_bound + self.edge_falloff):
                lower_curve = (self.upper_bound - self.edge_falloff)
                upper_curve = (self.upper_bound + self.edge_falloff)
                alpha = scurve3((control_value - lower_curve) / upper_curve - lower_curve)

                return linear_interp(self.sourceModules[1].get_value(x, y, z),
                    self.source0.get_value(x, y, z), alpha)
            else:
                return self.source0.get_value(x, y, z)
        else:
            if control_value < self.lower_bound or control_value > self.upper_bound:
                return self.source0.get_value(x, y, z)
            else:
                return self.source1.get_value(x, y, z)

class Spheres(NoiseModule):
    def __init__(self, frequency=1):
        self.frequency = frequency

    def get_value(self, x, y, z):
        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        center = (x*x + y*y + z*z)**(0.5)
        small = center - math.floor(center)
        large = 1 - small
        nearest = min(small, large)
        return 1 - (nearest*4)

class Terrace(NoiseModule):
    def __init__(self, invert_terraces=False):
        self.control_points = SortedList()
        self.invert_terraces = invert_terraces
        self.sourceModules = [None]

    def add_control_point(self, value):
        self.control_points.append(value)

    def get_value(self, x, y, z):
        assert (self.sourceModules[0] is not None)
        assert (len(self.control_points) > 1)

        value = self.sourceModules[0].get_value(x, y, z)

        for i, cp in enumerate(self.control_points):
            if value < cp:
                break

        index0 = clamp(i - 1, 0, len(self.control_points)-1)
        index1 = clamp(i, 0, len(self.control_points)-1)

        if index0 == index1:
            return self.control_points[index1]

        value0 = self.control_points[index0]
        value1 = self.control_points[index1]
        alpha = (value - value0) / (value1 - value0)

        if self.invert_terraces:
            alpha = 1 - alpha
            x = value0
            value0 = value1
            value1 = x

        alpha *= alpha

        return linear_interp(value0, value1, alpha)

class TranlatePoint(NoiseModule):
    def __init__(self, xtran, ytran, ztran):
        self.xtran = xtran
        self.ytran = ytran
        self.ztran = ztran
        self.sourceModules = [None]

    def get_value(self, x, y, z):
        assert (self.sourceModules[0] is not None)

        return self.sourceModules[0].get_value(x+xtran, y+ytran, z+ztran)

class Turbulence(NoiseModule):
    def __init__(self, frequency=1, power=1, roughness=3, seed=0):
        self.frequency = frequency
        self.power = power
        self.roughness = roughness
        self.seed = seed
        self.xdm = Perlin(frequency=frequency, octaves=roughness, seed=seed)
        self.ydm = Perlin(frequency=frequency, octaves=roughness, seed=seed)
        self.zdm = Perlin(frequency=frequency, octaves=roughness, seed=seed)

        self.sourceModules = [None]

    def get_value(self, x, y, z):
          x0 = x + (12414.0 / 65536.0)
          y0 = y + (65124.0 / 65536.0)
          z0 = z + (31337.0 / 65536.0)
          x1 = x + (26519.0 / 65536.0)
          y1 = y + (18128.0 / 65536.0)
          z1 = z + (60493.0 / 65536.0)
          x2 = x + (53820.0 / 65536.0)
          y2 = y + (11213.0 / 65536.0)
          z2 = z + (44845.0 / 65536.0)

          xDistort = x + (xdm.get_value(x0, y0, z0) * self.power)
          yDistort = y + (xdm.get_value(x1, y1, z1) * self.power)
          zDistort = z + (xdm.get_value(x2, y2, z2) * self.power)

          return self.sourceModules[0].get_value(xDistort, yDistort, zDistort)

class Voronoi(NoiseModule):
    def __init__(self, displacement=1, enable_distance=False, frequency=1, seed=0):
        self.displacement = displacement
        self.enable_distance = enable_distance
        self.frequency = frequency
        self.seed = seed

    def get_value(self, x, y, z):
        x *= self.frequency
        y *= self.frequency
        z *= self.frequency

        xInt = int(x) if (x > 0) else int(x) -1
        yInt = int(y) if (y > 0) else int(y) -1
        zInt = int(z) if (z > 0) else int(z) -1

        minDist = 2147483647.0
        xCan = 0
        yCan = 0
        zCan = 0

        for zCur in range(zInt-2, zInt+2):
            for yCur in range(yInt-2, yInt+2):
                for xCur in range(xInt-2, xInt+2):
                    xPos = xCur + value_noise_3d(xCur, yCur, zCur, self.seed)
                    yPos = yCur + value_noise_3d(xCur, yCur, zCur, self.seed+1)
                    zPos = zCur + value_noise_3d(xCur, yCur, zCur, self.seed+2)

                    xDist = xPos - x
                    yDist = yPos - y
                    zDist = zPos - z
                    dist = xDist * xDist + yDist * yDist + zDist * zDist

                    if dist < minDist:
                        minDist = dist
                        xCan = xPos
                        yCan = yPos
                        zCan = zPos
        value = 0

        if enable_distance:
            xDist = xCan - x
            yDist = yCan - y
            zDist = zCan - z

            value = (math.sqrt(xDist * xDist + yDist * yDist + zDist * zDist)) *\
                math.sqrt(3) - 1

        return value + (self.displacement * value_noise_3d(math.floor(xCan),
                                                math.floor(yCan),
                                                math.floor(zCan), 0))

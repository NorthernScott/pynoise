.. pynoise documentation master file, created by
   sphinx-quickstart on Thu Nov  5 09:24:31 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pynoise's documentation!
===================================

pynoise is a modular coherent noise generation system. With pynoise you can
combine various noise sources and operations to create interesting effects.

pynoise aims to be a complete, "batteries included" package, enabling a user to
generate, manipulate and render noise in a single package.

As an example, here is how to output a grayscale 512x512 image of some Perlin
noise::

    from pynoise.noisemodule import *
    from pynoise.noiseutil import *
    
    width, height = 512, 512
    perlin = Perlin()
    noise_map = noise_map_plane(width, height, 2, 6, 1, 5, perlin)
    gradient = grayscale_gradient()

    render = RenderImage()
    render.render(noise_map, width, height, 'perlin.png', gradient)

Features
--------

  * Base noise modules:

    * :py:class:`Perlin <pynoise.noisemodule.Perlin>`
    * :py:class:`Ridged Multi-Fractal <pynoise.noisemodule.RidgedMulti>`
    * :py:class:`Voronoi <pynoise.noisemodule.Voronoi>`
    * :py:class:`Sphere <pynoise.noisemodule.Spheres>`
    * :py:class:`Cylinders <pynoise.noisemodule.Cylinders>`
    * :py:class:`Checkerboard <pynoise.noisemodule.Checkerboard>`
    * :py:class:`Billow <pynoise.noisemodule.Billow>`
    * :py:class:`Const <pynoise.noisemodule.Const>`

  * Noise module modifiers:

    * :py:class:`Abs <pynoise.noisemodule.Abs>`
    * :py:class:`Add <pynoise.noisemodule.Add>`
    * :py:class:`Blend <pynoise.noisemodule.Blend>`
    * :py:class:`Clamp <pynoise.noisemodule.Clamp>`
    * :py:class:`Curve <pynoise.noisemodule.Curve>`
    * :py:class:`Displace <pynoise.noisemodule.Displace>`
    * :py:class:`Exponent <pynoise.noisemodule.Exponent>`
    * :py:class:`Invert <pynoise.noisemodule.Invert>`
    * :py:class:`Max <pynoise.noisemodule.Max>`
    * :py:class:`Min <pynoise.noisemodule.Min>`
    * :py:class:`Multiply <pynoise.noisemodule.Multiply>`
    * :py:class:`Power <pynoise.noisemodule.Power>`
    * :py:class:`RotatePoint <pynoise.noisemodule.RotatePoint>`
    * :py:class:`ScaleBias <pynoise.noisemodule.ScaleBias>`
    * :py:class:`ScalePoint <pynoise.noisemodule.ScalePoint>`
    * :py:class:`Select <pynoise.noisemodule.Select>`
    * :py:class:`Terrace <pynoise.noisemodule.Terrace>`
    * :py:class:`TranslatePoint <pynoise.noisemodule.TranslatePoint>`
    * :py:class:`Turbulence <pynoise.noisemodule.Turbulence>`

Installation
------------

Install the project by running:

    pip install pynoise

Contribute
----------

- Issue Tracker: https://gitlab.com/atrus6/pynoise/issues
- Source Code: https://gitlab.com/atrus6/pynoise

Support
-------

If you have any issues, please email me at tim at timchi dot me

License
-------

This is licensed under MPL 2.0

Tutorials
---------

 :doc:`tutorial1`

 :doc:`tutorial2`

 :doc:`tutorial3`

 :doc:`tutorial4`

 :doc:`tutorial5`

 :doc:`tutorial6`

 :doc:`tutorial7`

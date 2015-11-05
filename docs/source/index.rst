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

    from pynoise import noisemodule.*, noiseutil.*
    width, height = 512, 512
    perlin = Perlin()
    noise_map = noise_map_plane(lower_x=0, upper_x=1, lower_z=0, upper_z=1, width=width, height=height)
    gradient = grayscale_gradient()

    render = RenderImage()
    render.render(noise_map, width, height, 'perlin.png', gradient)

Features
--------

  * Base noise modules:

    * Perlin
    * Ridged Multi-Fractal
    * Voronoi
    * Sphere
    * Cylinders
    * Checkerboard
    * Billow
    * Const

  * Noise module modifiers:

    * Abs
    * Add
    * Blend
    * Clamp
    * Curve
    * Displace
    * Exponent
    * Invert
    * Max
    * Min
    * Multiply
    * Power
    * RotatePoint
    * ScaleBias
    * ScalePoint
    * Select
    * Terrace
    * TranslatePoint
    * Turbulence

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

Contents:

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Tutorial 2:
===========

What is coherent noise?
-----------------------

Coherent noise is a type of smooth pseudorandom noise. It has three important
properties.

1. The same input will give the same output.
2. A small change in input will produce a small change in output
3. A large change in input will produce a random change in output.

Which results in a repeatable output that produces smooth changes without sudden
jumps.

Generating a coherent noise value.
----------------------------------

With pynoise you generate coherent noise using **noise modules**. Pynoise provides
a large number of different modules, all of which are derived from
``pynoise.noisemodule.NoiseModule``.

For this tutorial, you will be using a
`Perlin noise <https://en.wikipedia.org/wiki/Perlin_noise>`_ module.

- To use this noise module, first activate your virtualenv::

    source venv/bin/activate

- Create a file called perlin.py::

    touch perlin.py

- Edit perlin.py so that it contains the following::

    from pynoise.noisemodule import Perlin

    perlin = Perlin()
    value = perlin.get_value(1.25, 0.75, 0.5)
    print(value)

- Run perlin.py and you should see...::

    0.45859943113378904

Changing the coordinates of the input value.
--------------------------------------------

If you slightly change the coordinates of the input value, the output will
slightly change.

In perlin.py change::

    value = perlin.get_value(1.25, 0.75, 0.5)
to::

   value = perlin.get_value(1.2501, 0.7501, 0.5001)

and you will have an output value of::

   0.45799774706607116

which is only marginally different from the original inputs you gave.

However, if you significantly change the input values, you will essentially
have new random values, for example, if you change::

   value = perlin.get_value(1.2501, 0.7501, 0.5001)
to::

   value = perlin.get_value(14.5, 20.25, 75.75)
you will get a value of::

   -0.5196255881079102

In ``tutorial3`` you will learn how to use these noise modules to create terrain
heightmaps.

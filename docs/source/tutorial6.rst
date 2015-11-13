Tutorial 6: Adding realism with turbulence
==========================================

This is a short tutorial, showing how we can increase realism by adding
small amounts of turbulence to out noisemap terrain.

Applying turbulence to noise modules
------------------------------------

The turbulence noise module randomly shifts the output of a source module.
It does this by using the output of three internal Perlin noise modules
for each x, y, and z coordinate.

For this tutorial, we will be using the code from tutorial 6. If you've
skipped over it, here it is.
::

    import sys
    sys.path.append('../pynoise')
    
    from pynoise.noisemodule import *
    from pynoise.noiseutil import *
    from pynoise.colors import Color
    
    rmf = RidgedMulti()
    billow = Billow(frequency=2)
    flatten = ScaleBias(source0=billow, scale=0.125, bias=-0.75)
    terrain_pick = Perlin(frequency=0.5, persistence=0.25)
    final = Select(source0=flatten, source1=rmf, source2=terrain_pick, lower_bound=0, upper_bound=1000, edge_falloff=0.125) 
    
    
    g = GradientColor()
    g.add_gradient_point(-1, Color(0.125, 0.627, 0))
    g.add_gradient_point(-0.25, Color(0.87, 0.87, 0))
    g.add_gradient_point(0.25, Color(0.5, 0.5, 0.5))
    g.add_gradient_point(1, Color(1, 1, 1))
    
    nm1 = noise_map_plane_gpu(256, 256, 6, 10, 1, 5, final)
    
    r = RenderImage(light_enabled=True, light_contrast=2, light_brightness=2)
    
    r.render(256, 256, nm1, 'tut5_5.png', g)

Instead of directly rendering `final` we will rename it to terrain and 
pass it to a Turbulence module.
::

    terrain = Select(source0=flatten, source1=rmf, source2=terrain_pick, lower_bound=0, upper_bound=1000, edge_falloff=0.125) 
    turbulence = Turbulence(source0=terrain, frequency=4, power=0.125)

    nm1 = noise_map_plane_gpu(256, 256, 6, 10, 1, 5, turbulence)

testing out output, we can see that we have less crazy ridges, our
cliff faces look like they've been eroded by time.

Modifying the power and frequency
---------------------------------

To modify the frequency, pass the frequency parameter, likewise for
power, pass the power parameter.

Frequency determines how often the coordinates change. Power determines
how much that change effects the original source. Determining the
'best' values is a subjective matter, mostly determined through
trial and error.

Now on to :doc:`tutorial7`

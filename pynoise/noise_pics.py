if __name__ == '__main__':
    import sys
    sys.path.append('/home/atrus/Copy/pynoise')

    from pynoise.noisemodule import Perlin, RidgedMulti, Billow, Voronoi
    from pynoise.noisemodule import Checkerboard, Cylinders, Spheres

    from pynoise.noiseutil import noise_map_plane, terrain_gradient, RenderImage

    nm_perlin = noise_map_plane(lower_x=2, upper_x=6, lower_z=1, upper_z=5, width=256, height=256, source=Perlin())
    nm_ridged = noise_map_plane(lower_x=2, upper_x=6, lower_z=1, upper_z=5, width=256, height=256, source=RidgedMulti())
    nm_billow = noise_map_plane(lower_x=2, upper_x=6, lower_z=1, upper_z=5, width=256, height=256, source=Billow())
    nm_voronoi = noise_map_plane(lower_x=2, upper_x=6, lower_z=1, upper_z=5, width=256, height=256, source=Voronoi())
    nm_checkerboard = noise_map_plane(lower_x=2, upper_x=6, lower_z=1, upper_z=5, width=256, height=256, source=Checkerboard())
    nm_cylinders = noise_map_plane(lower_x=2, upper_x=6, lower_z=1, upper_z=5, width=256, height=256, source=Cylinders())
    nm_spheres = noise_map_plane(lower_x=2, upper_x=6, lower_z=1, upper_z=5, width=256, height=256, source=Spheres())

    gradient = terrain_gradient()

    r = RenderImage()

    print('Perlin')
    r.render(nm_perlin, 'perlin.png', gradient)
    print('Ridged Multifractal')
    r.render(nm_ridged, 'ridged.png', gradient)
    print('Billow')
    r.render(nm_billow, 'billow.png', gradient)
    print('Voronoi')
    r.render(nm_voronoi, 'voronoi.png', gradient)
    print('Checkerboard')
    r.render(nm_checkerboard, 'checkerboard.png', gradient)
    print('Cylinders')
    r.render(nm_cylinders, 'cylinders.png', gradient)
    print('Spheres')
    r.render(nm_spheres, 'spheres.png', gradient)

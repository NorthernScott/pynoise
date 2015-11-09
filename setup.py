from setuptools import setup

setup(
    name = 'pynoise',
    include_package_data=True,
    packages = ['pynoise'],
    platforms='any',
    version = '2.2.1',
    description = 'An implementation of libnoise in python. Allows the creation of verious noise maps using a series of interconnected noise modules.',
    author = 'Tim Butram',
    author_email = 'tim@timchi.me',
    url = 'https://gitlab.com/atrus6/pynoise',
    download_url = 'https://gitlab.com/atrus6/pynoise/repository/archive.tar.gz',
    keywords = ['perlin', 'noise', 'procedural'],
    install_requires = ['sortedcontainers>=0.9.6', 'colormath', 'pillow', 'numpy', 'pyopencl']
)

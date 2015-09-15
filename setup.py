from distutils.core import setup

setup(
    name = 'pynoise',
    packages = ['pynoise'],
    version = '2.0.0a',
    description = 'An implementation of libnoise in python. Allows the creation of verious noise maps using a series of interconnected noise modules.',
    author = 'Tim Butram',
    author_email = 'tim@timchi.me',
    url = 'https://gitlab.com/atrus6/pynoise',
    download_url = 'https://gitlab.com/atrus6/pynoise/repository/archive.tar.gz?ref=1.0a',
    keywords = ['perlin', 'noise', 'procedural'],
    install_requires = ['sortedcontainers>=0.9.6', 'colormath', 'pillow', 'numpy']
)

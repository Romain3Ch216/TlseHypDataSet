from setuptools import setup

setup(
    name='TlseHypDataSet',
    version='0.0.1',
    description='A Python package to use and analyse the Toulouse Hyperspectral Data Set',
    url='https://github.com/Romain3Ch216/TlseHypDataSet',
    author='Romain Thoreau',
    author_email='romain.thoreau@onera.fr',
    packages=['TlseHypDataSet', 'TlseHypDataSet/utils',
              'TlseHypDataSet/dimension_reduction',
              'TlseHypDataSet/default_splits'],
    package_dir={'TlseHypDataSet': 'TlseHypDataSet'},
    include_package_data=True,
    install_requires=['numpy',
                      'torch',
                      'gdal',
                      'rasterio',
                      'protobuf',
                      'ortools',
                      'geopandas',
                      'torchvision',
                      'matplotlib'
                      ]
)

from setuptools import setup

setup(
    name='TlseHypDataSet',
    version='0.0.1',
    description='A Python package to use and analyse the Toulouse Hyperspectral Data Set',
    url='https://github.com/Romain3Ch216/TlseHypDataSet',
    author='Romain Thoreau',
    author_email='romain.thoreau@onera.fr',
    packages=['TlseHypDataSet', 'TlseHypDataSet/utils'],
    package_dir={'TlseHypDataSet': 'TlseHypDataSet'},
    include_package_data=True,
    install_requires=['numpy==1.19.5',
                      'torch==1.7.1',
                      'gdal==3.3.2',
                      'rasterio==1.2.10',
                      'ortools',
                      'geopandas==0.12.2',
                      'torchvision==0.8.2',
                      'matplotlib==3.3.2'
                      ]
)

from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()


setup(
    name='TlseHypDataSet',
    version='0.0.1',
    description='A Python package to use and analyse the Toulouse Hyperspectral Data Set',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/Romain3Ch216/TlseHypDataSet',
    author='Romain Thoreau',
    author_email='romain.thoreau@onera.fr',
    packages=['TlseHypDataSet', 'TlseHypDataSet/utils',
              'TlseHypDataSet/dimension_reduction',
              'TlseHypDataSet/default_splits',
              'TlseHypDataSet/ground_truth'],
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

from setuptools import setup
import os

version_path = os.path.join(os.path.dirname(__file__), 'loopy/VERSION')
if os.path.exists(version_path):
    with open(version_path) as f:
        VERSION = f.read().strip()
else:
    VERSION = '0.0.0'

setup(
    name='loopy',
    version='0.1',
    install_requires=[
        'networkx',
        'numpy',
        'scipy'
    ],
)

from setuptools import setup
import os

version_path = os.path.join(os.path.dirname(__file__), 'local_learning/VERSION')
if os.path.exists(version_path):
    with open(version_path) as f:
        VERSION = f.read().strip()
else:
    VERSION = '0.0'

setup(
    name='local_learning',
    version=VERSION,
    install_requires=[
        'numpy',
        'scipy'
    ],
)

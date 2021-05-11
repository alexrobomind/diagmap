#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(
    name="diagmap",
    version="1.2",
    description="Diagnostic mapping utility",
    author="Alexander Knieps",
    author_email="a.knieps@fz-juelich.de",
    url="https://github.com/alexrobomind/diagmap",
    py_modules=["diagmap"],
    package_dir={"": "src"},
    license="BSD-2-Clause",
    license_files=["LICENSE.txt"],
    install_requires=[
        "networkx>=2.4",
        "numpy>=1.18.5",
        "tqdm>=4.49.0",
        "scipy>=1.4.1",
    ],
)

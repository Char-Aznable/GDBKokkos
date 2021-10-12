#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2020 Char Aznable <aznable.char.0083@gmail.com>
# Description: setup for GDBKokkos release and install
#
# Distributed under terms of the 3-clause BSD license.
import setuptools

setuptools.setup(
    name="GDBKokkos",
    python_requires=">=3.8.10",
    version="0.1.0",
    description="GDB python modules for debugging Kokkos",
    long_description="see https://github.com/Char-Aznable/GDBKokkos",
    long_description_content_type="text/markdown",
    url="https://github.com/Char-Aznable/GDBKokkos.git",
    author="Char Aznable",
    author_email="aznable.char.0083@gmail.com",
    license="BSD 3-clause",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=["GDBKokkos"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
    ],
    extras_require = {
        "tests" : ["pytest", "pytest-cov", "codecov"]
    },
    data_files = [("", ["LICENSE"])]
)

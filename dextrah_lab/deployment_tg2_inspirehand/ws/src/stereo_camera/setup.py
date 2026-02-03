#!/usr/bin/env python3

from setuptools import find_packages, setup

try:
    from catkin_pkg.python_setup import generate_distutils_setup
except ImportError as exc:  # pragma: no cover - depends on ROS env
    raise ImportError("catkin_pkg is required to build this package.") from exc

setup_args = generate_distutils_setup(
    packages=find_packages(exclude=["tests", "tests.*"]),
)

setup(**setup_args)

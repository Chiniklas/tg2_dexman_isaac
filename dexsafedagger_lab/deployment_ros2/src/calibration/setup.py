#!/usr/bin/env python3

from setuptools import find_packages, setup


package_name = "calibration"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools", "numpy", "opencv-python", "torch", "warp-lang"],
    zip_safe=True,
    maintainer="tg2",
    maintainer_email="tg2@example.com",
    description="ROS 2 TG2 Inspirehand calibration nodes.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "april_tag_detector = calibration.april_tag_detector:main",
            "camera_calibration = calibration.camera_calibration:main",
        ],
    },
)

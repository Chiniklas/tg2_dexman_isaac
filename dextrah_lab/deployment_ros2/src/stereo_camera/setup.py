#!/usr/bin/env python3

from glob import glob

from setuptools import find_packages, setup


package_name = "stereo_camera"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md", "requirements.txt"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    package_data={
        package_name: [
            "cameras/config/*.yaml",
            "StereoCameraCalibration/*.pdf",
        ],
    },
    install_requires=["setuptools", "numpy", "opencv-python", "PyYAML"],
    zip_safe=True,
    maintainer="tg2",
    maintainer_email="tg2@example.com",
    description="ROS 2 stereo camera utilities for TG2 Inspirehand.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "stereo_ros_publisher = stereo_camera.stereo_ros_publisher_node:main",
        ],
    },
)

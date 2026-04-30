#!/usr/bin/env python3

from glob import glob

from setuptools import find_packages, setup


package_name = "policy_inference_stereo_transformer"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "opencv-python", "PyYAML", "torch"],
    zip_safe=True,
    maintainer="tg2",
    maintainer_email="tg2@example.com",
    description="ROS 2 stereo transformer policy inference bridge for TG2 deployment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_inference_stereo_transformer_node = "
            "policy_inference_stereo_transformer.policy_inference_stereo_transformer_node:main",
        ],
    },
)

#!/usr/bin/env python3

from setuptools import find_packages, setup


package_name = "inference_offline"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "h5py", "numpy", "PyYAML", "torch"],
    zip_safe=True,
    maintainer="tg2",
    maintainer_email="tg2@example.com",
    description="Isaac Lab offline replay entry point packaged for the ROS 2 deployment workspace.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_inference_offline = inference_offline.policy_inference_offline:main",
        ],
    },
)

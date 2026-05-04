# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Dextrah in Isaac Lab package setuptools."""

#
# References:
# * https://setuptools.pypa.io/en/latest/setuptools.html#setup-cfg-only-projects

# Third Party
from setuptools import find_packages, setup

STEREO_CAMERA_ROOT = "dextrah_lab/deployment_tg2_inspirehand/Stereo_camera_ov9732"
STEREO_CAMERA_PKG = f"{STEREO_CAMERA_ROOT}/stereo_camera"

base_packages = find_packages(exclude=("tests",))
stereo_packages = find_packages(STEREO_CAMERA_ROOT, exclude=("tests",))

setup(
    name="dextrah_lab",
    packages=base_packages + stereo_packages,
    package_dir={
        "": ".",
        "stereo_camera": STEREO_CAMERA_PKG,
    },
    install_requires=[
        "numpy>=1.23.5,<2.0.0",
        "networkx==3.3",
        "opencv-python",
        "pillow==11.3.0",
        "PyYAML",
        "warp-lang>=1.5.0",
    ],
)

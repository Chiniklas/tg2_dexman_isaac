from setuptools import find_packages, setup


setup(
    name="tg2_lab",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "mujoco>=3.1.6",
        "numpy>=1.23.5,<2.0.0",
        "PyYAML",
        "torch>=2.4.0",
        "warp-lang>=1.5.0",
    ],
)

from setuptools import setup


package_name = "feedback_control_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/feedback_control_bridge.launch.py"]),
        (f"share/{package_name}", ["README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Tiangong Robotics",
    maintainer_email="robotics@tiangong.example",
    description="ROS 2 direct joint command and status bridge for Tiangong arm and hand control.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "feedback_control_bridge = feedback_control_bridge.feedback_control_bridge_node:main",
        ],
    },
)

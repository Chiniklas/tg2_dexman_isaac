# Noetic on 20.04 with RViz/MoveIt and OpenGL userspace
FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# Core build + ROS tooling + GUI/OpenGL libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep python3-catkin-tools python3-h5py build-essential cmake git \
    ros-noetic-moveit \
    ros-noetic-ros-control ros-noetic-ros-controllers \
    ros-noetic-robot-state-publisher ros-noetic-joint-state-publisher-gui \
    ros-noetic-diagnostic-updater ros-noetic-diagnostics \
    mesa-utils libgl1-mesa-dri libglu1-mesa libglfw3 libxkbcommon0 \
 && rm -rf /var/lib/apt/lists/*

# (Optional) MuJoCo & analysis stack; comment out if not needed
# RUN pip3 install --no-cache-dir mujoco==3.1.6 numpy scipy matplotlib pandas pyyaml transforms3d

# Convenience: source ROS and (if present) the workspace overlay
RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc && \
    echo 'if [ -f /tiangong_infra_ws/ws/devel/setup.bash ]; then source /tiangong_infra_ws/ws/devel/setup.bash; elif [ -f /tiangong_infra_ws/devel/setup.bash ]; then source /tiangong_infra_ws/devel/setup.bash; fi' >> /root/.bashrc

WORKDIR /tiangong_infra_ws
CMD ["/bin/bash"]

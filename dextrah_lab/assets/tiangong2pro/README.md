# Tiangong2Pro Asset Folder

This folder is the non-destructive asset-style view of the ROS2 package at:

`dextrah_lab/assets/tiangong2pro_urdf_ros2/tiangong2pro_urdf`

Current layout:

- `meshes/`: symlink to the ROS2 package mesh directory
- `urdf/`: a single copied URDF asset for this bundle
- `xml/`: a standalone MuJoCo XML for display/import
- `usd/`: reserved for Isaac/Omniverse USD exports

Nothing in the original ROS2 package is deleted or moved.

MuJoCo path for this asset:

- URDF: `urdf/tiangong2.0_pro_with_hands.urdf`
- standalone XML: `xml/tiangong2.0_pro_with_hands.xml`

The XML is checked in directly. There is no repo-side build script for this
asset bundle.

This MuJoCo asset intentionally uses only the full
`tiangong2.0_pro_with_hands.urdf` variant and does not keep the
collision-stripped `*_kinematic.urdf` copy in this folder.

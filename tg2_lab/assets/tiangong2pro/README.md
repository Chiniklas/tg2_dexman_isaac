# Tiangong2Pro Asset Folder

This folder is the non-destructive asset-style view of the ROS2 package at:

`tg2_lab/assets/tiangong2pro_urdf_ros2/tiangong2pro_urdf`

Current layout:

- `meshes/`: symlink to the ROS2 package mesh directory
- `urdf/`: a single copied URDF asset for this bundle
- `xml/`: a standalone MuJoCo XML for display/import
- `usd/`: Isaac/Omniverse USD exports

The active Isaac Lab robot config uses the fixed-head USD reimport:

- `usd/tiangong2.0_pro_with_hands_half_body_fixed_head/tiangong2.0_pro_with_hands_half_body_fixed_head.usd`

That USD is generated from `urdf/tiangong2.0_pro_with_hands_half_body_fixed_head.urdf`
with self-collision enabled to match the KUKA-SHARPA reference task more closely.

Nothing in the original ROS2 package is deleted or moved.

MuJoCo path for this asset:

- URDF: `urdf/tiangong2.0_pro_with_hands.urdf`
- standalone XML: `xml/tiangong2.0_pro_with_hands.xml`

The XML is checked in directly. There is no repo-side build script for this
asset bundle.

This MuJoCo asset intentionally uses only the full
`tiangong2.0_pro_with_hands.urdf` variant and does not keep the
collision-stripped `*_kinematic.urdf` copy in this folder.

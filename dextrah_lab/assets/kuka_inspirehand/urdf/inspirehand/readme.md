this folder is for inspirehand_RH56DFX

note: the simpliest link graph is this:

world - base    - 3 - 4 (index)
                - 13 - 14 (middle)
                - 23 - 24 (ring)
                - 33 - 34 (little)
                - 43 - 44 - 47 - 48 (thumb)


right hand is simplified and can be run in mujoco (check ./mjcf/handright9253_clean.mjcf). also in ./urdf/handright_9253_simplified.urdf, I have deleted all the unnecessary links and joints. It can be run in pybullet and moveit.

left hand is not modified.

TODO:
1- simplify lefthand
2- add usd version for isaacsim
3- attach hand to body
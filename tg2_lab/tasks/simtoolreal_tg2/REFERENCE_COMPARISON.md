# TG2 vs. KUKA-SHARPA SimToolReal Reference

This note is the starting point for future agents investigating why the
reference KUKA + SHARPA policy grasps the DexToolBench `claw_hammer`, while the
TG2 + InspireHand policy reaches and sometimes lifts it but does not maintain a
stable grasp.

Reference implementation:

- `reference/simtoolreal_isaacsim/simtoolreal_lab/tasks/simtoolreal_sharpa`
- `reference/simtoolreal_isaacsim/simtoolreal_lab/assets/kuka_sharpa_forge`

Active TG2 implementation:

- `tg2_lab/tasks/simtoolreal_tg2`
- `tg2_lab/assets/tiangong2pro/robot.py`

## Main conclusion

The checked-in PPO/SAPO hyperparameters are nearly identical. The most likely
blockers are the retargeted robot-task interface: hand controllability, reward
point calibration, actuator/contact dynamics, and the initial robot/object
geometry. The KUKA reference is also a single-arm system, so single-arm control
by itself does not explain the difference. Do not start by tuning PPO unless
the physical and geometric checks below pass.

## Material differences

| Area | KUKA + SHARPA reference | TG2 + InspireHand |
| --- | --- | --- |
| Policy actions | 29: 7 arm + 22 hand | 13: 7 arm + 6 hand |
| Hand control | Mostly independent hand joints | Six commands expanded to twelve joints through fixed mimic ratios |
| Actor observations | 140 | 110: full state of 7 arm + 12 hand joints |
| Privileged critic states | 162 | 132 |
| Palm reward offset | `(0.0, -0.02, 0.16)` | `(0.0, 0.0, 0.0)` |
| Fingertip reward offsets | `(0.02, 0.002, 0.0)` for each fingertip | All zero |
| Hand dynamics | SHARPA-specific calibrated values | SHARPA-like gains/limits copied onto InspireHand joints |
| Object reset | +/-10 cm XY, +/-2 cm Z, random rotation | +/-5 cm XY, fixed Z and rotation |
| Robot reset | Joint position and velocity noise | Fixed joint state |
| Table reset | +/-1 cm height | Fixed height |
| Goal reset | Random 3D pose and rotation, then delta/coin-flip updates | Vertical goal 25--35 cm above the initial object, with unchanged rotation |
| Sim-to-real corruption | Delays, state noise, and post-lift disturbances enabled | Enabled to match the successful 2026-06-19 run |
| Scene cloning | `replicate_physics=True`, spacing 1.2 | `replicate_physics=False`, spacing 2.0 |

The hammer mass, hammer scale, simulation rate, episode duration, reward
weights, SAPO exploration blocks, PPO learning rate, horizon, clipping, LSTM,
and MLP sizes are otherwise aligned closely.

## Why these differences matter

1. **Reduced hand authority.** SHARPA exposes 22 hand actions. TG2 exposes six
   hand actions and imposes fixed distal-joint synergies. The available synergy
   may not produce opposition and force closure around the hammer handle.
2. **Possibly incorrect reward geometry.** TG2 currently measures raw palm and
   fingertip body origins. The reference applies calibrated offsets. The policy
   can therefore improve its distance reward without moving the actual contact
   surfaces into a useful grasp.
3. **Uncalibrated dynamics.** TG2 uses dictionaries named `_SHARPA_*` for
   stiffness, damping, effort/velocity limits, armature, and friction. Similar
   numbers do not guarantee similar behavior because the InspireHand link
   masses, transmissions, joint axes, limits, and collision geometry differ.
4. **Weak grasp supervision.** In addition to fingertip-distance and object-height
   shaping, TG2 now gives each finger one 100-point bonus when its distal link
   contacts the object while its fingertip is within 2 cm of the hammer's
   oriented bounds. It also requires at least one hand-object contact when
   crossing the lift threshold. These gates still do not require opposing
   contacts, handle containment, or stable grasp duration. SHARPA's richer hand
   can discover a grasp under weaker shaping; the coupled TG2 hand may still
   settle on reaching, pushing, and occasional unstable lifting.
5. **Noise sensitivity.** Both active recipes enable observation delay, action
   delay, object-state noise, and post-lift disturbances. The successful
   2026-06-19 TG2 run used these settings, while a later deterministic run did
   not cross the grasp-and-lift threshold; multiple seeds are still needed to
   distinguish a systematic noise benefit from exploration variance.

The easier TG2 reset distribution and vertical goal should not make initial
grasp discovery harder. Failure under these easier conditions is additional
evidence that the hand interface or contact dynamics are the first place to
look. Existing curves showing roughly 10--12 cm of occasional lift also suggest
that gross arm reachability is present, while grasp retention is not.

## Scripted initial-pose calibration

The palm-up pose was read from Isaac Sim's Physics Inspector on 2026-07-06.
Right-arm joint positions for the demonstrated grasp, in degrees, are shoulder
`(0, -5.73, 0)`, elbow `(-88.1, -93.1)`, and wrist `(0, 0)`. The inspector showed a shoulder-roll
target of `0` degrees beyond its `-5.73` degree upper limit, so the test uses
the valid limit. Kinematic inspection identifies palm local `-X` as its upward
support normal in this pose. The default hammer offset is consequently
the demonstrated robot-root-local translation `(0.4518, -0.32205, 0.09672)`
with RPY `(0, 0, 90)` degrees. This manually verified pose establishes that the
hand asset and hammer contact model can achieve a physical grasp.

The automated dynamic test also passed with this pose on 2026-07-06. After one
initial placement (with no subsequent teleporting), the hand retained the
hammer for 300/300 static-hold and arm-wave steps with `0.0002 m` maximum
palm-relative drift and no termination. Treat hand-asset grasp feasibility as
validated; remaining RL failures should be investigated in reset placement,
action exploration/synergy discovery, observations, rewards, and curriculum.

The demonstrated successful hand pose uses `63` degrees for index, middle,
ring, and little proximal joints, `59.2` degrees for `thumb_joint_0`, and
`14.1` degrees for `thumb_joint_1`. In the six-dimensional normalized policy
action this is approximately `(1, 1, 1, 1, 0.6294, -0.0156)`. Do not use all
ones: that drives `thumb_joint_1` to `28.648` degrees and can push the hammer
out instead of opposing the fingers.

## Recommended investigation order

1. Run `tests/test_scripted_hammer_closure.py` in the active task folder. It
   initializes the arm palm-up, places the hammer in the hand, closes, releases,
   and waves the arm. Confirm that the current six-dimensional hand command can
   physically retain the hammer without RL.
2. Visualize and calibrate `palm_offset` and every `fingertip_offset` against the
   actual collision/contact surfaces and the hammer handle.
3. Inspect finger/hammer collision shapes, contact reports, friction, self-
   collision pairs, and whether at least two opposing contacts occur.
4. Tune InspireHand-specific stiffness, damping, effort limits, armature, and
   friction instead of assuming SHARPA values transfer.
5. Keep the restored observation/action delay, pose noise, and post-lift
   disturbance settings fixed while comparing multiple seeds; ablate them
   separately rather than changing all corruption sources at once.
6. If the scripted grasp cannot achieve force closure, expose more independent
   finger joints or define a hammer-specific synergy before training again.
7. If physics and control work but RL still pushes the hammer, add grasp-stage
   signals such as opposing-contact count, handle containment, palm-handle
   distance, and stable-grasp duration.
8. Restore reset randomization, full 3D goal sampling, delays, noise, and
   disturbances incrementally after deterministic success.
9. Tune PPO/SAPO only after the above checks pass.

## Reference-parity gaps already visible in code

- TG2's `_reset_goals()` does not use the configured target volume,
  `delta_goal_distance`, or `delta_rotation_degrees`; it samples only vertical
  goals above the initial object pose.
- TG2 reset logic does not currently apply reference-style object rotation,
  object Z, robot DOF, velocity, or table-height randomization.
- The TG2 main recipe uses no object-scale randomization.
- TG2 has not yet validated InspireHand actuator/contact parity against the
  successful SHARPA setup.

Treat these as separate stages: first establish deterministic TG2 grasp
feasibility, then restore reference parity and robustness.

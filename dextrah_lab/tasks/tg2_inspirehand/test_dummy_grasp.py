"""Dummy grasp sanity check for the TG2-Inspirehand.

Boot Isaac Sim/Kit first via AppLauncher so carb/USD are available.
Spawns the arm+hand with ground plane and dome light, then holds a nominal pose.
"""

import argparse

from isaaclab.app import AppLauncher


def main():
    """Spawn the robot in a simple scene and hold a nominal joint configuration."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch Kit before importing anything that needs carb/USD.
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import math
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import build_simulation_context
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from dextrah_lab.assets.tg2_inspirehand.tg2_inspirehand import TG2_INSPIREHAND_CFG

    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except Exception:
        plt = None
        Slider = None

    # Nominal joint angles for a reachable tabletop pose; fingers open.
    init_joint_pos = {
        "shoulder_pitch_r_joint": -0.89,
        "shoulder_roll_r_joint": -0.93200582,
        "shoulder_yaw_r_joint": 0.31590459,
        "elbow_pitch_r_joint": -1.60221225,
        "elbow_yaw_r_joint": -0.02094395,
        "wrist_pitch_r_joint": -0.17453293,
        "wrist_roll_r_joint": -0.18675023,
        "index_joint_0": 0.25,
        "little_joint_0": 0.25,
        "middle_joint_0": 0.25,
        "ring_joint_0": 0.25,
        "thumb_joint_0": 0.5,
        "index_joint_1": 0.386,
        "little_joint_1": 0.386,
        "middle_joint_1": 0.386,
        "ring_joint_1": 0.386,
        "thumb_joint_1": 0.1,
        "thumb_joint_2": 0.2,
        "thumb_joint_3": 0.4,
    }
    nominal_joint_pos = init_joint_pos
    sine_amp_rad = 0.2  # default 0.1
    sine_freq_hz = 0.5  # default 0.5
    report_every_s = 1.0  # print joint states at this interval
    plot_every_s = 0.01  # update plot at this interval
    subplot_height = 0.12  # subplot height in figure coordinates
    subplot_gap = 0.02  # vertical gap between subplots

    @configclass
    class DummyGraspSceneCfg(InteractiveSceneCfg):
        """Simple scene with ground, lighting, and the robot."""

        # Ground plane and lighting
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=4000.0, color=(0.9, 0.9, 0.9)),
        )

        # Robot at nominal pose
        robot = TG2_INSPIREHAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TG2_INSPIREHAND_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.25),
                rot=(0.0, 0.0, 0.0, 1.0),
                joint_pos=init_joint_pos,
            ),
        )

    # Build the scene
    scene_cfg = DummyGraspSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    device = getattr(args_cli, "device", "cuda:0")
    sim_dt = 1.0 / 120.0

    # Spin up simulation context before building the scene so SimulationContext.instance() is valid.
    with build_simulation_context(device=device, dt=sim_dt, add_ground_plane=False, add_lighting=False) as sim:
        scene = InteractiveScene(scene_cfg)

        # Start physics (this triggers asset initialization/actuators), then reset scene.
        sim.reset()
        scene.reset()

        robot = scene.articulations["robot"]
        joint_pos = robot.data.joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.joint_vel)
        for idx, name in enumerate(robot.joint_names):
            if name in nominal_joint_pos:
                joint_pos[:, idx] = nominal_joint_pos[name]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)

        print("[INFO] Dummy grasp scene running. Close the window to exit.")
        t = 0.0
        step_count = 0
        report_every_steps = max(1, int(report_every_s / sim_dt))
        plot_every_steps = max(1, int(plot_every_s / sim_dt))
        num_joints = len(robot.joint_names)
        plot_times = []
        plot_desired = [[] for _ in range(num_joints)]
        plot_observed = [[] for _ in range(num_joints)]
        plot_fig = None
        plot_axes = []
        plot_lines_obs = []
        plot_lines_des = []
        scroll_slider = None
        if plt is None:
            print("[WARN] matplotlib unavailable; skipping plot rendering.")
        else:
            plt.ion()
            plot_fig = plt.figure(figsize=(10, 6))
            plot_left = 0.08
            plot_width = 0.82
            viewport_bottom = 0.06
            viewport_top_margin = 0.08
            viewport_height = 1.0 - viewport_bottom - viewport_top_margin
            content_height = num_joints * (subplot_height + subplot_gap) - subplot_gap
            max_scroll = max(0.0, content_height - viewport_height)
            scroll_val = 0.0
            for name in robot.joint_names:
                ax = plot_fig.add_axes([plot_left, viewport_bottom, plot_width, subplot_height])
                color = ax._get_lines.get_next_color()
                line_obs, = ax.plot([], [], color=color)
                line_des, = ax.plot([], [], color=color, linestyle="--", alpha=0.6)
                plot_axes.append(ax)
                plot_lines_obs.append(line_obs)
                plot_lines_des.append(line_des)
                ax.set_ylabel("rad")
                ax.set_title(f"{name} (solid=observed, dashed=desired)", fontsize=8, pad=2)
                ax.grid(True, alpha=0.3)
            plot_axes[-1].set_xlabel("time (s)")

            def layout_axes(scroll_val: float) -> None:
                scroll_val = max(0.0, min(float(scroll_val), max_scroll))
                for i, ax in enumerate(plot_axes):
                    y_top = content_height - i * (subplot_height + subplot_gap)
                    y0 = viewport_bottom + y_top - subplot_height - scroll_val
                    ax.set_position([plot_left, y0, plot_width, subplot_height])
                    visible = (y0 + subplot_height) >= viewport_bottom and y0 <= (viewport_bottom + viewport_height)
                    ax.set_visible(visible)
                plot_fig.canvas.draw_idle()

            if max_scroll > 0.0:
                if Slider is None:
                    print("[WARN] matplotlib slider unavailable; use mouse wheel to scroll.")

                    def on_scroll(event) -> None:
                        nonlocal scroll_val
                        step = subplot_height + subplot_gap
                        delta = -step if event.button == "up" else step
                        scroll_val = min(max(scroll_val + delta, 0.0), max_scroll)
                        layout_axes(scroll_val)

                    plot_fig.canvas.mpl_connect("scroll_event", on_scroll)
                else:
                    scroll_ax = plot_fig.add_axes([0.93, viewport_bottom, 0.02, viewport_height])
                    scroll_slider = Slider(
                        scroll_ax,
                        "",
                        0.0,
                        max_scroll,
                        valinit=0.0,
                        orientation="vertical",
                    )
                    scroll_ax.set_xticks([])
                    scroll_ax.set_yticks([])
                    scroll_slider.on_changed(layout_axes)

                    def on_scroll(event) -> None:
                        step = subplot_height + subplot_gap
                        delta = -step if event.button == "up" else step
                        new_val = min(max(scroll_slider.val + delta, 0.0), max_scroll)
                        scroll_slider.set_val(new_val)

                    plot_fig.canvas.mpl_connect("scroll_event", on_scroll)

            layout_axes(0.0)
            plt.show(block=False)
        while simulation_app.is_running():
            # Apply a simple sine wave around the nominal pose.
            phase = 2.0 * math.pi * sine_freq_hz * t
            for idx, name in enumerate(robot.joint_names):
                if name in nominal_joint_pos:
                    joint_pos[:, idx] = nominal_joint_pos[name] + sine_amp_rad * math.sin(phase)
            robot.set_joint_position_target(joint_pos)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(dt=sim_dt)
            if step_count % report_every_steps == 0:
                joint_pos_report = robot.data.joint_pos[0].tolist()
                joint_vel_report = robot.data.joint_vel[0].tolist()
                print("[JOINT STATE]")
                for name, pos, vel in zip(robot.joint_names, joint_pos_report, joint_vel_report):
                    print(f"  {name}: pos={pos:.4f} rad, vel={vel:.4f} rad/s")
            if plot_fig is not None and step_count % plot_every_steps == 0:
                plot_times.append(t)
                desired_snapshot = joint_pos[0].tolist()
                observed_snapshot = robot.data.joint_pos[0].tolist()
                for idx in range(num_joints):
                    plot_desired[idx].append(desired_snapshot[idx])
                    plot_observed[idx].append(observed_snapshot[idx])
                    plot_lines_obs[idx].set_data(plot_times, plot_observed[idx])
                    plot_lines_des[idx].set_data(plot_times, plot_desired[idx])
                for ax in plot_axes:
                    ax.relim()
                    ax.autoscale_view()
                plot_fig.canvas.draw_idle()
                plot_fig.canvas.flush_events()
                plt.pause(0.001)
            t += sim_dt
            step_count += 1
        if plot_fig is not None:
            plt.close(plot_fig)


if __name__ == "__main__":
    main()

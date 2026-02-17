#!/usr/bin/env python3
"""Pure-python AprilTag detection demo (no ROS).

This test mirrors the reference detector logic as closely as possible:
- detect tags with cv2.aruco
- estimate pose using solvePnP on tag corners
- report tag pose and camera pose w.r.t. tag

Difference from reference: camera source is stereo_camera left input.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


DICT_MAP = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

FLIP_MAP = {"none": None, "vertical": 0, "horizontal": 1, "both": -1}
FAMILY_TO_DICT_NAME = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}


def _find_ws_src(start: Path) -> Path:
    """Find <ws>/src so stereo_camera imports work from anywhere."""
    marker_rel = Path("stereo_camera/stereo_camera/cameras/ov9732_camera.py")
    for candidate in [start, *start.parents]:
        direct = candidate / marker_rel
        if direct.exists():
            return candidate
        via_src = candidate / "src" / marker_rel
        if via_src.exists():
            return candidate / "src"
    raise FileNotFoundError(
        f"Could not locate stereo_camera package by searching from {start}. "
        f"Expected {marker_rel}"
    )


def _load_intrinsics_from_npz(npz_path: Path, use_right_camera: bool) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Intrinsic file not found: {npz_path}")
    calib = np.load(npz_path)
    if use_right_camera:
        key_k_candidates = ("k2", "k")
        key_d_candidates = ("d2", "d")
    else:
        key_k_candidates = ("k1", "k")
        key_d_candidates = ("d1", "d")

    key_k = next((k for k in key_k_candidates if k in calib), None)
    key_d = next((k for k in key_d_candidates if k in calib), None)
    if key_k is None or key_d is None:
        raise KeyError(
            f"Could not find intrinsic keys in {npz_path}. "
            f"Expected one of {key_k_candidates} and {key_d_candidates}."
        )
    k = np.array(calib[key_k], dtype=np.float64).reshape(3, 3)
    d = np.array(calib[key_d], dtype=np.float64).reshape(-1)
    return k, d


def _build_detector(tag_family: str):
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[tag_family])
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        return detector.detectMarkers
    return lambda img: cv2.aruco.detectMarkers(  # type: ignore[return-value]
        img, aruco_dict, parameters=cv2.aruco.DetectorParameters_create()
    )


def _flip_point(x: int, y: int, width: int, height: int, flip_code: Optional[int]) -> tuple[int, int]:
    if flip_code is None:
        return x, y
    if flip_code == 0:  # vertical
        return x, height - 1 - y
    if flip_code == 1:  # horizontal
        return width - 1 - x, y
    if flip_code == -1:  # both
        return width - 1 - x, height - 1 - y
    return x, y


def _rotation_matrix_to_rpy_deg(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in degrees from a 3x3 rotation matrix (XYZ intrinsic)."""
    sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure-python left-camera AprilTag detection with Matplotlib visualization."
    )
    parser.add_argument("--left-config", default="ov9732_L", help="Left camera config name (YAML)")
    parser.add_argument("--device", default=None, help="Override camera device path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="none",
        help="Display-only flip applied after detection/pose estimation",
    )

    parser.add_argument("--tag-family", choices=sorted(DICT_MAP.keys()), default="tag25h9")
    parser.add_argument("--tag-id", type=int, default=-1, help="Detect all IDs if < 0")
    parser.add_argument("--tag-size", type=float, default=0.10, help="Tag size in meters")

    parser.add_argument(
        "--intrinsics-npz",
        default="",
        help=(
            "Optional calibration .npz (for pose axes). "
            "If empty, pose axes are disabled and only marker detection is shown."
        ),
    )
    parser.add_argument(
        "--intrinsics-camera",
        choices=["left", "right", "auto"],
        default="left",
        help="Which intrinsic keys to use from .npz",
    )
    parser.add_argument("--preview-scale", type=float, default=1.0, help="Matplotlib image scale")
    return parser.parse_args()


def main() -> int:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module not available. Install opencv-contrib-python.")

    args = parse_args()
    ws_src = _find_ws_src(Path(__file__).resolve().parent)
    if str(ws_src) not in sys.path:
        sys.path.insert(0, str(ws_src))

    from stereo_camera.cameras.ov9732_camera import Ov9732Camera, load_camera_config

    cfg = load_camera_config(args.left_config)
    overrides = {
        "device": args.device or cfg.get("device"),
        "width": args.width or cfg.get("width"),
        "height": args.height or cfg.get("height"),
        "fps": args.fps or cfg.get("fps"),
    }
    cam = Ov9732Camera.from_config(args.left_config, overrides=overrides)

    cam_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    if args.intrinsics_npz:
        intr_path = Path(args.intrinsics_npz)
        use_right = args.intrinsics_camera == "right"
        if args.intrinsics_camera == "auto":
            use_right = False
        cam_matrix, dist_coeffs = _load_intrinsics_from_npz(intr_path, use_right_camera=use_right)
        print(f"Loaded intrinsics from {intr_path}")

    detect = _build_detector(args.tag_family)
    flip_code = FLIP_MAP[args.flip]
    half = args.tag_size / 2.0
    obj_points = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float32,
    )
    dictionary_name = FAMILY_TO_DICT_NAME.get(args.tag_family, "DICT_APRILTAG_36h11")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8 * args.preview_scale, 6 * args.preview_scale))
    img_artist = None
    ax.set_axis_off()

    frame_count = 0
    t_prev = time.time()
    fps_val = 0.0
    last_log_time: dict[int, float] = {}

    try:
        cam.start()
        while plt.fignum_exists(fig.number):
            ok, frame = cam.read()
            if not ok or frame is None:
                plt.pause(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect(gray)

            selected_corners = []
            selected_ids = []
            if ids is not None and len(ids) > 0:
                for i, tag_id_arr in enumerate(ids.flatten()):
                    tag_id = int(tag_id_arr)
                    if args.tag_id >= 0 and tag_id != args.tag_id:
                        continue
                    selected_corners.append(corners[i])
                    selected_ids.append(tag_id)

            vis = frame.copy()
            pending_text: list[tuple[str, int, int, tuple[int, int, int]]] = []
            if selected_corners:
                cv2.aruco.drawDetectedMarkers(
                    vis,
                    selected_corners,
                    np.array(selected_ids, dtype=np.int32).reshape(-1, 1),
                )

                if cam_matrix is not None and dist_coeffs is not None:
                    for i, tag_id in enumerate(selected_ids):
                        img_pts = selected_corners[i].reshape(4, 2).astype(np.float32)
                        ok, rvec, tvec = cv2.solvePnP(
                            obj_points,
                            img_pts,
                            cam_matrix,
                            dist_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                        if not ok:
                            continue
                        cv2.drawFrameAxes(
                            vis,
                            cam_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            args.tag_size * 0.5,
                        )
                        position = tvec.flatten()
                        dist_m = float(np.linalg.norm(position))
                        rot, _ = cv2.Rodrigues(rvec)
                        roll_deg, pitch_deg, yaw_deg = _rotation_matrix_to_rpy_deg(rot)
                        camera_position_tag = (-rot.T @ tvec).flatten()
                        cxy = selected_corners[i].reshape(-1, 2).mean(axis=0).astype(int)
                        pending_text.append(
                            (
                                f"id={tag_id} d={dist_m:.2f}m z={position[2]:.2f}m",
                                int(cxy[0]) - 40,
                                int(cxy[1]) - 10,
                                (0, 255, 0),
                            )
                        )
                        now = time.time()
                        if now - last_log_time.get(tag_id, 0.0) >= 0.5:
                            print(
                                f"[{dictionary_name}] tag {tag_id} pos(x,y,z)=("
                                f"{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})m "
                                f"rpy_deg=({roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f}) "
                                f"distance={dist_m:.3f}m camera_wrt_tag=("
                                f"{camera_position_tag[0]:.3f}, {camera_position_tag[1]:.3f}, "
                                f"{camera_position_tag[2]:.3f})m"
                            )
                            last_log_time[tag_id] = now

            frame_count += 1
            t_now = time.time()
            dt = t_now - t_prev
            if dt >= 0.5:
                fps_val = frame_count / dt
                frame_count = 0
                t_prev = t_now

            status = (
                f"{args.tag_family} ({dictionary_name})"
                + (f":{args.tag_id}" if args.tag_id >= 0 else ":*")
                + f" | detections={len(selected_corners)} | fps={fps_val:.1f}"
            )
            vis_display = cv2.flip(vis, flip_code) if flip_code is not None else vis
            h, w = vis_display.shape[:2]
            for text, x, y, color in pending_text:
                tx, ty = _flip_point(x, y, w, h, flip_code)
                cv2.putText(
                    vis_display,
                    text,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                vis_display,
                status,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            vis_rgb = cv2.cvtColor(vis_display, cv2.COLOR_BGR2RGB)
            if img_artist is None:
                img_artist = ax.imshow(vis_rgb)
            else:
                img_artist.set_data(vis_rgb)
            ax.set_title("Pure Python AprilTag Detection (close window to quit)")
            fig.canvas.draw_idle()
            plt.pause(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        plt.ioff()
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Stereo camera calibration helpers.

Core flow:
1) Stream synchronized frames from two capture objects.
2) Collect N chessboard pairs when the user presses 'c'.
3) Solve individual intrinsics, then stereo extrinsics (R, T).
4) Save results to .npz files and provide lightweight load helpers.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple, List

from stereo_camera.utils.capture_two_stream import DualCameraReader, FLIP_MAP


class CameraInterface:
    """Minimal interface required from camera objects."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self) -> Tuple[bool, np.ndarray]: ...
    def release(self) -> None: ...


def _build_object_points(board_size: Tuple[int, int], square_size: float) -> np.ndarray:
    """3-D points for the checkerboard (Z=0 plane)."""
    objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def stereo_calibrate_camera(
    camera_left: CameraInterface,
    camera_right: CameraInterface,
    camera_name: str,
    square_size_mm: float = 1.0,
    board_size: Tuple[int, int] = (9, 6),
    frames_to_capture: int = 50,
    display_scale: float = 0.6,
    save_dir: str | Path = ".",
    fix_intrinsic: bool = True,
    max_fails: int = 5,
    reconnect_wait: float = 1.0,
    flip: str = "vertical",
) -> Path:
    """
    Run stereo calibration on two cameras showing a chessboard.

    Keys:
      press 'c' -> store a synchronized pair
      press 'x' -> abort without writing

    Returns path to the saved stereo .npz file.
    """
    if flip not in FLIP_MAP:
        raise ValueError("flip must be one of: none, vertical, horizontal, both")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stereo_path = save_dir / f"{camera_name}.npz"
    c1_path = save_dir / f"{camera_name}c1.npz"
    c2_path = save_dir / f"{camera_name}c2.npz"

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsic else 0

    objp = _build_object_points(board_size, square_size_mm)
    objpoints: List[np.ndarray] = []
    imgpoints_l: List[np.ndarray] = []
    imgpoints_r: List[np.ndarray] = []
    frames_l: List[np.ndarray] = []
    frames_r: List[np.ndarray] = []

    reader = DualCameraReader([camera_left, camera_right], flip=flip, max_fails=max_fails, reconnect_wait=reconnect_wait)
    reader.labels = ["L", "R"]
    reader.start()

    print("Align checkerboard; 'c' capture pair, 'x' abort.")
    captured = 0

    try:
        while captured < frames_to_capture:
            frames = reader.read_pair()
            if frames is None:
                cv2.waitKey(10)
                continue

            frame_l, frame_r = frames

            preview = cv2.hconcat(
                [
                    cv2.resize(frame_l, (0, 0), fx=display_scale, fy=display_scale),
                    cv2.resize(frame_r, (0, 0), fx=display_scale, fy=display_scale),
                ]
            )
            cv2.imshow("left | right", preview)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("c"):
                frames_l.append(frame_l.copy())
                frames_r.append(frame_r.copy())
                captured += 1
                print(f"{captured}/{frames_to_capture} captured")
            elif key == ord("x"):
                print("Aborting capture.")
                return stereo_path
    finally:
        cv2.destroyAllWindows()
        reader.stop()

    for img_l, img_r in zip(frames_l, frames_r):
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        ret_l, corners_l = cv2.findChessboardCorners(
            gray_l, board_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        ret_r, corners_r = cv2.findChessboardCorners(
            gray_r, board_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not (ret_l and ret_r):
            continue

        objpoints.append(objp)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

    if not objpoints:
        raise RuntimeError("No valid chessboard pairs collected; calibration aborted.")

    h, w = frames_l[0].shape[:2]
    _, k1, d1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_l, (w, h), None, None)
    _, k2, d2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_r, (w, h), None, None)

    np.savez(c1_path, k=k1, d=d1, r=rvecs1, t=tvecs1)
    np.savez(c2_path, k=k2, d=d2, r=rvecs2, t=tvecs2)

    _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, k1, d1, k2, d2, (w, h), criteria=criteria, flags=flags
    )
    np.savez(stereo_path, k1=k1, d1=d1, k2=k2, d2=d2, R=R, T=T)

    return stereo_path


def load_stereo_parameters(file_path: str | Path) -> Tuple[np.ndarray, ...]:
    """Load combined stereo parameters from an .npz file."""
    data = np.load(file_path)
    return data["k1"], data["d1"], data["k2"], data["d2"], data["R"], data["T"]


def load_single_camera_parameters(file_path: str | Path) -> Tuple[np.ndarray, ...]:
    """Load intrinsics/distortion/rvecs/tvecs for one camera."""
    data = np.load(file_path)
    return data["k"], data["d"], data["r"], data["t"]

"""
Lightweight camera wrapper for OV9732/CSI sensors on Jetson or UVC devices.

Implements the minimal interface expected by `StereoCameraCalibration`:
start(), stop(), read(), release(). Frames are pulled on a background thread
so callers get the latest frame quickly.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import yaml


def gstreamer_pipeline(
    sensor_id: int = 0,
    sensor_mode: int = 3,
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 1280,
    display_height: int = 720,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def resolve_log_path(log_name: str) -> Path:
    """
    Build a timestamped log path under logs/<timestamp>/<log_name>.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / ts / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_camera_config(name: str | Path) -> Dict[str, Any]:
    """
    Load a camera config YAML by name or explicit path.
    - If `name` is a Path, load it directly.
    - If `name` is a string without extension, load CONFIG_DIR/<name>.yaml.
    """
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{name}.yaml"
    elif not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Camera config not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    return data


class Ov9732Camera:
    """
    Threaded OpenCV capture wrapper.

    Use either GStreamer (nvarguscamerasrc) or a V4L2 device path.
    """

    def __init__(
        self,
        *,
        device: str = "/dev/video0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_method: int = 0,
        use_gstreamer: bool = False,
        fourcc: str = "MJPG",
        gstreamer_kwargs: Optional[dict] = None,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.use_gstreamer = use_gstreamer
        self.fourcc = fourcc
        self.gstreamer_kwargs = gstreamer_kwargs or {}

        self.video_capture: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.grabbed: bool = False
        self.read_thread: Optional[threading.Thread] = None
        self.read_lock = threading.Lock()
        self.running = False

        # stats
        self.total_frames = 0
        self.dropped_frames = 0
        self.window_times = []
        self.window_size = 100
        self.last_frame_time: Optional[float] = None
        self.last_stats_time: Optional[float] = None
        self.start_time: Optional[float] = None

    @classmethod
    def from_config(cls, name: str = "ov9732_L", overrides: Optional[Dict[str, Any]] = None):
        cfg = load_camera_config(name)
        if overrides:
            cfg.update({k: v for k, v in overrides.items() if v is not None})
        return cls(
            device=cfg.get("device", "/dev/video0"),
            width=cfg.get("width", 1280),
            height=cfg.get("height", 720),
            fps=cfg.get("fps", 30),
            flip_method=cfg.get("flip_method", 0),
            use_gstreamer=cfg.get("use_gstreamer", False),
            fourcc=cfg.get("fourcc", "MJPG"),
            gstreamer_kwargs=cfg.get("gstreamer_kwargs", {}),
        )

    # Open the underlying capture
    def open(self, **override) -> None:
        if self.video_capture is not None:
            return

        cfg = {
            "device": override.get("device", self.device),
            "width": override.get("width", self.width),
            "height": override.get("height", self.height),
            "fps": override.get("fps", self.fps),
            "flip_method": override.get("flip_method", self.flip_method),
        }
        use_gst = override.get("use_gstreamer", self.use_gstreamer)
        gst_kwargs = {**self.gstreamer_kwargs, **override.get("gstreamer_kwargs", {})}

        if use_gst:
            pipeline = gstreamer_pipeline(
                sensor_id=gst_kwargs.get("sensor_id", 0),
                sensor_mode=gst_kwargs.get("sensor_mode", 3),
                capture_width=gst_kwargs.get("capture_width", cfg["width"]),
                capture_height=gst_kwargs.get("capture_height", cfg["height"]),
                display_width=gst_kwargs.get("display_width", cfg["width"]),
                display_height=gst_kwargs.get("display_height", cfg["height"]),
                framerate=gst_kwargs.get("framerate", cfg["fps"]),
                flip_method=cfg["flip_method"],
            )
            logging.info("Opening camera via GStreamer: %s", pipeline)
            self.video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # Use V4L2 only (per project preference).
            logging.info("Opening camera via V4L2: %s", cfg["device"])
            self.video_capture = cv2.VideoCapture(cfg["device"], cv2.CAP_V4L2)
            fourcc = cv2.VideoWriter_fourcc(*str(self.fourcc))
            self.video_capture.set(cv2.CAP_PROP_FOURCC, fourcc)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
            self.video_capture.set(cv2.CAP_PROP_FPS, cfg["fps"])

        if self.video_capture is None or not self.video_capture.isOpened():
            self.video_capture = None
            raise RuntimeError("Unable to open camera")

        self.grabbed, self.frame = self.video_capture.read()

    def start(self):
        if self.running:
            return self
        if self.video_capture is None:
            self.open()
        if self.start_time is None:
            self.start_time = time.monotonic()
        self.running = True
        self.read_thread = threading.Thread(target=self._update_camera, daemon=True)
        self.read_thread.start()
        return self

    def _update_camera(self):
        while self.running and self.video_capture is not None:
            grabbed, frame = self.video_capture.read()
            now = time.monotonic()
            if not grabbed or frame is None:
                self.dropped_frames += 1
                time.sleep(0.005)
                continue
            self.total_frames += 1
            if self.last_frame_time is not None:
                dt = now - self.last_frame_time
                self.window_times.append(dt)
                if len(self.window_times) > self.window_size:
                    self.window_times.pop(0)
            self.last_frame_time = now
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self) -> Tuple[bool, np.ndarray]:
        with self.read_lock:
            frame = None if self.frame is None else self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.running = False
        if self.read_thread is not None:
            self.read_thread.join()
            self.read_thread = None

    def release(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.stop()
        self.window_times.clear()
        self.last_frame_time = None
        self.last_stats_time = None
        self.start_time = None
        self.total_frames = 0
        self.dropped_frames = 0

    def save_video(self, path, codec: str = "MJPG", fps: Optional[float] = None):
        """
        Create a cv2.VideoWriter configured to match the current capture settings.

        Parameters
        ----------
        path : str or Path
            Output file path.
        codec : str, optional
            FourCC string (default "MJPG").
        fps : float, optional
            Frames per second; defaults to the camera's configured fps.

        Returns
        -------
        cv2.VideoWriter
            Opened writer (caller is responsible for writer.release()).
        """
        fourcc_val = cv2.VideoWriter_fourcc(*codec)
        target_fps = float(self.fps if fps is None else fps)
        writer = cv2.VideoWriter(str(path), fourcc_val, target_fps, (int(self.width), int(self.height)))
        return writer

    def log_stats(self, interval: float = 10.0, label: str = "") -> bool:
        """
        Log capture stats if `interval` seconds have elapsed since last log.

        Returns True if a log line was emitted.
        """
        now = time.monotonic()
        if self.last_stats_time is None:
            self.last_stats_time = now
            return False
        if now - self.last_stats_time < interval:
            return False

        avg_dt = sum(self.window_times) / len(self.window_times) if self.window_times else 0.0
        avg_fps = (1.0 / avg_dt) if avg_dt > 0 else 0.0
        uptime = now - self.start_time if self.start_time is not None else 0.0
        logging.info(
            "[%s] uptime=%.1fs total_frames=%d dropped=%d avg_fps=%.2f last_dt=%.3fs",
            label or self.device,
            uptime,
            self.total_frames,
            self.dropped_frames,
            avg_fps,
            avg_dt,
        )
        self.last_stats_time = now
        return True


__all__ = ["Ov9732Camera", "gstreamer_pipeline"]

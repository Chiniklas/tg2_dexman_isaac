from __future__ import annotations


def initialize_warp(device: str = "cpu") -> None:
    """Initialize Warp runtime and optionally select device."""
    try:
        import warp as wp
    except ImportError as exc:
        raise ImportError("Warp is required for calibration kinematics but is not installed.") from exc

    wp.init()

    # Keep this defensive because Warp APIs differ across versions.
    if not device:
        return
    if hasattr(wp, "set_device"):
        try:
            wp.set_device(device)
        except Exception:
            pass
    elif hasattr(wp, "get_device"):
        try:
            wp.get_device(device)
        except Exception:
            pass

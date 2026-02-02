"""
Test/CLI wrapper for the live stereo depth viewer.
Delegates to stereo_camera.utils.capture_stereo_stream.main so tests/ can be run with -m.
"""

from __future__ import annotations

import sys

from stereo_camera.utils.capture_stereo_stream import main


if __name__ == "__main__":
    sys.exit(main())

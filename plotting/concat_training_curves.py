#!/usr/bin/env python3
"""Concatenate the configured training-curve metric plots horizontally."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


DEFAULT_INPUT_ROOT = Path("/home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/plotting/plots")
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_ROOT / "training_curves_concat.png"

IMAGE_ORDER = (
    "comparison_avg_lift_success.png",
    "comparison_beta.png",
    "comparison_unsafe_episode_rate.png",
)

BACKGROUND = (255, 255, 255)
PADDING = 16


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concatenate the configured training curve PNGs horizontally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing the training curve PNGs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output image path for the concatenated result.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    input_root = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    image_paths = [input_root / name for name in IMAGE_ORDER]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected training curve images:\n" + "\n".join(f"  - {path}" for path in missing)
        )

    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        max_height = max(image.height for image in images)
        total_width = sum(image.width for image in images) + PADDING * (len(images) + 1)

        canvas = Image.new("RGB", (total_width, max_height + 2 * PADDING), BACKGROUND)
        x_offset = PADDING
        for image in images:
            y_offset = PADDING + (max_height - image.height) // 2
            canvas.paste(image, (x_offset, y_offset))
            x_offset += image.width + PADDING

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    finally:
        for image in images:
            image.close()

    print(f"Saved concatenated image to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

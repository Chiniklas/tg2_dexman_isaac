#!/usr/bin/env python3
"""Concatenate per-object metric plots into a labeled grid image."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_INPUT_ROOT = Path("/home/chizhang/projects/dextrah/tg2_dexman_isaac/plotting/plots")
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_ROOT / "object_metric_grid.png"

ROW_SPECS = (
    ("unsafe_episode_rate", "Unsafe Episode Rate"),
    ("harmful_collision", "Harmful Collision"),
    ("object_out_of_bound", "Object Out Of Bound"),
    ("palm_flipped", "Palm Flipped"),
    ("hand_too_far", "Hand Too Far"),
)

OBJECT_ALIASES = {
    "windcart": "1m0lvpzs",
    # "boot": "2kp2e9k7",
    # "toolbox": "2oiqpnts",
    "cow": "z73ltdbb",
}

BACKGROUND = (255, 255, 255)
TEXT_COLOR = (32, 32, 32)
GRID_COLOR = (225, 225, 225)
PADDING = 16
HEADER_HEIGHT = 0
ROW_LABEL_WIDTH = 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concatenate per-object plot PNGs into a single array image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root plots directory containing one subdirectory per object.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output image path for the combined grid.",
    )
    return parser


def _discover_object_dirs(input_root: Path) -> list[Path]:
    object_dirs = []
    for child in sorted(p for p in input_root.iterdir() if p.is_dir()):
        if any((child / f"{metric_key}.png").exists() for metric_key, _ in ROW_SPECS):
            object_dirs.append(child)
    return object_dirs


def _resolve_object_dirs(input_root: Path) -> list[Path]:
    discovered = _discover_object_dirs(input_root)
    dir_map = {path.name.lower(): path for path in discovered}

    resolved = []
    for object_id in OBJECT_ALIASES.values():
        object_dir = dir_map.get(object_id.lower())
        if object_dir is None:
            continue
        resolved.append(object_dir)
    if not resolved:
        configured = " ".join(OBJECT_ALIASES.values())
        available = " ".join(path.name for path in discovered)
        raise ValueError(
            "None of the configured object rows were found. "
            f"Configured object ids: {configured}. Available directories: {available}"
        )
    return resolved


def _measure_cell_size(object_dirs: list[Path]) -> tuple[int, int]:
    max_width = 0
    max_height = 0
    for object_dir in object_dirs:
        for metric_key, _ in ROW_SPECS:
            image_path = object_dir / f"{metric_key}.png"
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                max_width = max(max_width, image.width)
                max_height = max(max_height, image.height)
    if max_width == 0 or max_height == 0:
        raise ValueError("No plot PNGs were found for the requested grid.")
    return max_width, max_height


def _fit_image(image: Image.Image, cell_size: tuple[int, int]) -> Image.Image:
    cell_width, cell_height = cell_size
    fitted = image.copy()
    fitted.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", cell_size, BACKGROUND)
    offset_x = (cell_width - fitted.width) // 2
    offset_y = (cell_height - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    return canvas


def main() -> int:
    args = _build_parser().parse_args()
    input_root = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    object_dirs = _resolve_object_dirs(input_root)
    if not object_dirs:
        raise ValueError(f"No object subdirectories with metric PNGs found under: {input_root}")

    cell_width, cell_height = _measure_cell_size(object_dirs)
    cols = len(ROW_SPECS)
    rows = len(object_dirs)
    canvas_width = ROW_LABEL_WIDTH + cols * cell_width + (cols + 2) * PADDING
    canvas_height = HEADER_HEIGHT + rows * cell_height + (rows + 2) * PADDING

    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for row_idx, object_dir in enumerate(object_dirs):
        for col_idx, (metric_key, _) in enumerate(ROW_SPECS):
            image_path = object_dir / f"{metric_key}.png"
            cell_x = ROW_LABEL_WIDTH + (col_idx + 1) * PADDING + col_idx * cell_width
            cell_y = HEADER_HEIGHT + (row_idx + 1) * PADDING + row_idx * cell_height
            if image_path.exists():
                with Image.open(image_path) as image:
                    fitted = _fit_image(image.convert("RGB"), (cell_width, cell_height))
                canvas.paste(fitted, (cell_x, cell_y))
            draw.rectangle(
                (cell_x, cell_y, cell_x + cell_width, cell_y + cell_height),
                outline=GRID_COLOR,
                width=1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved concatenated image to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

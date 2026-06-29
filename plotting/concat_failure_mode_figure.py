#!/usr/bin/env python3
"""Build the final failure-mode figure from a saved header strip and metric grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCES_DIR = SCRIPT_DIR / "sources"
DEFAULT_HEADER_NAME = "failure_mode_header.png"
DEFAULT_GRID_PATH = SCRIPT_DIR / "plots" / "object_metric_grid.png"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "plots" / "failure_mode.png"

BACKGROUND = (255, 255, 255)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stack the saved failure-mode visual header above the object metric grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=DEFAULT_SOURCES_DIR,
        help=f"Directory containing {DEFAULT_HEADER_NAME}.",
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=None,
        help=f"Optional explicit path to the cropped header image. Defaults to --sources/{DEFAULT_HEADER_NAME}.",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=DEFAULT_GRID_PATH,
        help="Existing object_metric_grid.png produced by concat_pics.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path for the final stacked figure.",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF output path for the same final stacked figure.",
    )
    return parser


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image.copy()
    height = int(round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> int:
    args = _build_parser().parse_args()
    sources_dir = args.sources.expanduser().resolve()
    header_path = (
        args.header.expanduser().resolve()
        if args.header is not None
        else sources_dir / DEFAULT_HEADER_NAME
    )
    grid_path = args.grid.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not header_path.is_file():
        raise FileNotFoundError(f"Header image does not exist: {header_path}")
    if not grid_path.is_file():
        raise FileNotFoundError(f"Object metric grid does not exist: {grid_path}")

    with Image.open(header_path) as header_image:
        header = header_image.convert("RGB")
    with Image.open(grid_path) as grid_image:
        grid = grid_image.convert("RGB")

    header = _resize_to_width(header, grid.width)
    final = Image.new("RGB", (grid.width, header.height + grid.height), BACKGROUND)
    final.paste(header, (0, 0))
    final.paste(grid, (0, header.height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(output_path)
    print(f"Saved final failure-mode figure to: {output_path}")

    if args.pdf_output is not None:
        pdf_output_path = args.pdf_output.expanduser().resolve()
        pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
        final.save(pdf_output_path, "PDF", resolution=300.0)
        print(f"Saved final failure-mode PDF to: {pdf_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

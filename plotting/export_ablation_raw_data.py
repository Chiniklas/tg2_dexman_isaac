#!/usr/bin/env python3
"""Export compact, plot-ready ablation scalars inside each run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from ablation_raw_data import export_ablation_dataset


SCRIPT_DIR = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess and export only the TensorBoard scalars used by the plots."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "config.yaml",
        help="YAML file containing the ablation runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="raw_data directory inside one timestamped plotting output.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    export_ablation_dataset(
        config_path=args.config,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

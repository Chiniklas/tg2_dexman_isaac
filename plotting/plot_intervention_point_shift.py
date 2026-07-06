#!/usr/bin/env python3
"""Plot the VLM-guided intervention-threshold trajectory in two dimensions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "dexsafedagger_lab" / "distillation" / "runs"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "plots" / "intervention_point_shift.png"
DEFAULT_FIGSIZE = (6.4, 6.4)


@dataclass(frozen=True)
class ThresholdPoint:
    step: int
    l2_threshold: float
    success_threshold: float
    label: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot applied L2 and success-predictor threshold updates from a "
            "VLM threshold-advisor JSONL log."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to vlm_threshold_advisor.jsonl. If omitted, use the most "
            "recent advisor log under the distillation runs directory."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Root searched when --input is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_FIGSIZE,
        help="Figure size in inches; the square default matches the training panels.",
    )
    parser.add_argument(
        "--annotate-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Label applied points with their online step.",
    )
    parser.add_argument(
        "--hypothetical-success-end",
        type=float,
        default=None,
        help=(
            "Replace logged success thresholds with a monotonic illustrative "
            "chain from the initial value to this final value."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Export a compact inset version without a title, axis labels, "
            "legend, or colorbar."
        ),
    )
    return parser


def _resolve_input(input_path: Path | None, runs_root: Path) -> Path:
    if input_path is not None:
        resolved = input_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"VLM advisor log does not exist: {resolved}")
        return resolved

    runs_root = runs_root.expanduser().resolve()
    candidates = list(runs_root.glob("*/vlm_threshold_advisor.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No vlm_threshold_advisor.jsonl files found under: {runs_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_trajectory(path: Path) -> list[ThresholdPoint]:
    records = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if record.get("status") != "success":
            continue
        applied = record.get("applied") or {}
        if applied.get("l2_threshold") is None or applied.get("success_threshold") is None:
            continue
        records.append(record)

    if not records:
        raise ValueError(f"No successful applied threshold updates found in: {path}")

    first_stats = records[0].get("stats") or {}
    if first_stats.get("l2_threshold") is None or first_stats.get("success_threshold") is None:
        raise ValueError("First successful record lacks the initial threshold pair.")

    points = [
        ThresholdPoint(
            step=0,
            l2_threshold=float(first_stats["l2_threshold"]),
            success_threshold=float(first_stats["success_threshold"]),
            label="initial",
        )
    ]
    for record in records:
        applied = record["applied"]
        step = int(record.get("step", 0))
        points.append(
            ThresholdPoint(
                step=step,
                l2_threshold=float(applied["l2_threshold"]),
                success_threshold=float(applied["success_threshold"]),
                label=f"{step:,}",
            )
        )
    return points


def _padded_limits(values: np.ndarray, *, minimum_pad: float) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    pad = max(minimum_pad, span * 0.15)
    return low - pad, high + pad


def _with_hypothetical_success_chain(
    points: list[ThresholdPoint],
    final_threshold: float,
) -> list[ThresholdPoint]:
    initial_threshold = float(points[0].success_threshold)
    final_threshold = float(final_threshold)
    if np.isclose(initial_threshold, final_threshold):
        raise ValueError(
            "Hypothetical final success threshold must differ from the initial value."
        )
    values = np.linspace(initial_threshold, final_threshold, num=len(points))
    return [
        ThresholdPoint(
            step=point.step,
            l2_threshold=point.l2_threshold,
            success_threshold=float(value),
            label=point.label,
        )
        for point, value in zip(points, values)
    ]


def _plot_trajectory(
    points: list[ThresholdPoint],
    *,
    annotate_steps: bool,
    clean: bool,
    figsize: tuple[float, float],
):
    l2_values = np.asarray([point.l2_threshold for point in points], dtype=float)
    success_values = np.asarray([point.success_threshold for point in points], dtype=float)
    steps = np.asarray([point.step for point in points], dtype=float)
    success_is_constant = bool(np.ptp(success_values) < 1e-10)

    fig, axis = plt.subplots(figsize=figsize)
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=float(np.min(steps)), vmax=max(1.0, float(np.max(steps))))

    for source, target in zip(points[:-1], points[1:]):
        dx = target.l2_threshold - source.l2_threshold
        dy = target.success_threshold - source.success_threshold
        if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
            continue
        axis.annotate(
            "",
            xy=(target.l2_threshold, target.success_threshold),
            xytext=(source.l2_threshold, source.success_threshold),
            arrowprops={
                "arrowstyle": "-|>",
                "color": cmap(norm(target.step)),
                "linewidth": 2.0,
                "mutation_scale": 14,
                "shrinkA": 5,
                "shrinkB": 5,
            },
            zorder=2,
        )

    scatter = axis.scatter(
        l2_values,
        success_values,
        c=steps,
        cmap=cmap,
        norm=norm,
        s=58,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    axis.scatter(
        [points[0].l2_threshold],
        [points[0].success_threshold],
        marker="o",
        s=110,
        facecolors="none",
        edgecolors="#20242a",
        linewidths=1.8,
        zorder=4,
    )
    axis.scatter(
        [points[-1].l2_threshold],
        [points[-1].success_threshold],
        marker="*",
        s=180,
        color=cmap(norm(points[-1].step)),
        edgecolors="#20242a",
        linewidths=0.8,
        zorder=4,
    )

    if annotate_steps:
        for index, point in enumerate(points[1:], start=1):
            offset_y = (10 if index % 2 else -16) if success_is_constant else 10
            axis.annotate(
                f"{point.step // 1000}k",
                (point.l2_threshold, point.success_threshold),
                xytext=(0, offset_y),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color="#30343b",
            )

    if success_is_constant:
        axis.text(
            0.02,
            0.06,
            f"Success threshold unchanged at {success_values[0]:.6f}",
            transform=axis.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
        )

    axis.set_xlim(*_padded_limits(l2_values, minimum_pad=0.005))
    axis.set_ylim(*_padded_limits(success_values, minimum_pad=0.01))
    if not clean:
        axis.set_xlabel("Teacher–Student Action Disagreement Threshold", fontsize=13)
        axis.set_ylabel("Risk Predictor Success Threshold", fontsize=13)
        axis.set_title("VLM-Guided Intervention Point Shift", fontsize=16)
    axis.grid(True, color="#d9d9d9", alpha=0.8, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=11)

    if not clean:
        colorbar = fig.colorbar(scatter, ax=axis, pad=0.02)
        colorbar.set_label("Online Step", fontsize=11)
        legend_handles = [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                   markeredgecolor="#20242a", markeredgewidth=1.8, markersize=8, label="Initial"),
            Line2D([], [], marker="*", linestyle="none", color="#20242a",
                   markersize=11, label="Final"),
        ]
        axis.legend(handles=legend_handles, loc="best", frameon=True)
    fig.tight_layout()
    return fig


def main() -> int:
    args = _build_parser().parse_args()
    input_path = _resolve_input(args.input, args.runs_root)
    output_path = args.output.expanduser().resolve()
    points = _read_trajectory(input_path)
    hypothetical = args.hypothetical_success_end is not None
    if hypothetical:
        points = _with_hypothetical_success_chain(
            points,
            args.hypothetical_success_end,
        )
    figure = _plot_trajectory(
        points,
        annotate_steps=args.annotate_steps,
        clean=args.clean,
        figsize=tuple(args.figsize),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Loaded {len(points) - 1} applied threshold updates from: {input_path}")
    print(f"Saved intervention-point trajectory to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

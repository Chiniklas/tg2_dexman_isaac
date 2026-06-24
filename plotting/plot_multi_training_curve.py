#!/usr/bin/env python3
"""Plot comparison figures from TensorBoard runs listed in a YAML config."""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter


DEFAULT_TAG_PRIORITY = [
    "beta",
    "train/avg/unsafe_episode_rate",
    "eval/avg/lift_success",
]

DEFAULT_TAG_FILENAMES = {
    "beta": "beta",
    "train/avg/unsafe_episode_rate": "unsafe_episode_rate",
    "eval/avg/lift_success": "avg_lift_success",
}

DEFAULT_TAG_TITLES = {
    "beta": "Comparison Intervention Rate Beta",
    "train/avg/unsafe_episode_rate": "Comparison Unsafe Episode Rate",
    "eval/avg/lift_success": "Comparison Student Policy Lift Success",
}

TAG_PLOT_OVERRIDES = {
    "beta": {
        "smoothing": 0.95,
        "downsample": 20,
        "show_raw": False,
        "legend_loc": "upper right",
        "band_scale": 1.0,
        "band_alpha": 0.14,
        "band_smoothing": 0.999,
    },
    "train/avg/unsafe_episode_rate": {
        "smoothing": 0.9,
        "downsample": 20,
        "show_raw": False,
        "legend_loc": "upper right",
        "band_scale": 1.0,
        "band_alpha": 0.14,
        "band_smoothing": 0.999,
    },
    "eval/avg/lift_success": {
        "smoothing": 0.75,
        "downsample": 1,
        "show_raw": True,
        "legend_loc": "lower right",
        "band_scale": 1.0,
        "band_alpha": 0.14,
    },
}

PLOT_TAGS = list(DEFAULT_TAG_PRIORITY)
X_AXIS = "iteration"
SMOOTHING = 0.75
DOWNSAMPLE = 5
SHOW_RAW = False
MIN_STEP = None
MAX_STEP = None
SHOW_FIGURES = False
CUSTOM_TITLE = None
FIGSIZE = (6.5, 6.5)
DPI = 180
LINEWIDTH = 1.8 # default 1.6
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "plots" / "comparison.png"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
TRAIN_NUM_ENVS = 32
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16


@dataclass(frozen=True)
class RunSpec:
    label: str
    source_path: Path
    event_files: tuple[Path, ...]


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    value: float
    wall_time: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot comparison figures from TensorBoard runs listed in YAML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML file containing the training runs to compare.",
    )
    parser.add_argument(
        "--scalar-cache",
        type=Path,
        default=None,
        help="Optional shared scalar cache produced by prepare_scalar_cache.py.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Override the default scalar tags to plot.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("step", "iteration", "time", "index"),
        default=X_AXIS,
        help="Horizontal axis for plotting.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=SMOOTHING,
        help="EMA smoothing factor in [0, 1). 0 disables smoothing.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=DOWNSAMPLE,
        help="Keep every Nth point for plotting. 1 disables downsampling.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=MIN_STEP,
        help="Drop points before this global step.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=MAX_STEP,
        help="Drop points after this global step.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Base output path for saved figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=SHOW_FIGURES,
        help="Open matplotlib windows after saving figures.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=CUSTOM_TITLE,
        help="Optional shared title override for non-default tags.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=FIGSIZE,
        help="Matplotlib figure size in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help="Figure DPI when saving.",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=LINEWIDTH,
        help="Line width for each run curve.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        default=SHOW_RAW,
        help="Overlay the downsampled raw trace under the smoothed curve.",
    )
    return parser


def _import_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard event reading requires the 'tensorboard' package. "
            "Install it in the Python environment used to run this script."
        ) from exc
    return event_accumulator


def _find_event_files(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("events.out.tfevents*") if p.is_file())


def _load_run_specs(config_path: Path) -> list[tuple[str, str]]:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Plot config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    raw_runs = config.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"Expected a non-empty 'runs' list in: {config_path}")

    run_specs = []
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise ValueError(f"Run entry {index} must be a mapping in: {config_path}")
        if not raw_run.get("enabled", True):
            continue

        label = str(raw_run.get("label", "")).strip()
        raw_path = str(raw_run.get("path", "")).strip()
        if not label or not raw_path:
            raise ValueError(
                f"Enabled run entry {index} requires non-empty 'label' and 'path' values."
            )

        run_path = Path(raw_path).expanduser()
        if not run_path.is_absolute():
            run_path = config_path.parent / run_path
        run_specs.append((label, str(run_path.resolve())))

    if not run_specs:
        raise ValueError(f"No enabled runs were found in: {config_path}")
    return run_specs


def _load_run_colors(config_path: Path) -> dict[str, str]:
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    colors = {}
    for raw_run in config.get("runs", []):
        if not isinstance(raw_run, dict) or not raw_run.get("enabled", True):
            continue
        label = str(raw_run.get("label", "")).strip()
        color = str(raw_run.get("color", "")).strip()
        if label and color:
            colors[label] = color
    return colors


def _logical_run_root(event_file: Path) -> Path:
    parent = event_file.parent
    if parent.name == "summaries":
        return parent.parent
    return parent


def _resolve_runs(run_specs: list[tuple[str, str]]) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for label, raw_path in run_specs:
        requested_path = Path(raw_path).expanduser().resolve()
        event_files = _find_event_files(requested_path)
        if not event_files:
            raise FileNotFoundError(f"No TensorBoard event files found under: {requested_path}")

        grouped_files: dict[Path, list[Path]] = defaultdict(list)
        for event_file in event_files:
            grouped_files[_logical_run_root(event_file)].append(event_file)

        if len(grouped_files) != 1:
            run_roots = "\n".join(f"  - {run_root}" for run_root in sorted(grouped_files))
            raise ValueError(
                f"Expected exactly one run under '{requested_path}', but found {len(grouped_files)}:\n{run_roots}"
            )

        run_root, files = next(iter(sorted(grouped_files.items())))
        runs.append(
            RunSpec(
                label=label,
                source_path=run_root,
                event_files=tuple(sorted(files)),
            )
        )
    return runs


def _load_run_scalars(run: RunSpec) -> dict[str, list[ScalarPoint]]:
    event_accumulator = _import_event_accumulator()
    size_guidance = {event_accumulator.SCALARS: 0}
    scalars_by_tag: dict[str, list[ScalarPoint]] = defaultdict(list)

    for event_file in run.event_files:
        size_mb = event_file.stat().st_size / (1024.0 * 1024.0)
        print(
            f"[training curves] Loading {run.label}: {event_file.name} "
            f"({size_mb:.1f} MB)...",
            flush=True,
        )
        accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance=size_guidance)
        accumulator.Reload()
        print(
            f"[training curves] Loaded {run.label}; extracting scalar tags...",
            flush=True,
        )
        for tag in accumulator.Tags().get("scalars", []):
            scalars_by_tag[tag].extend(
                ScalarPoint(step=int(s.step), value=float(s.value), wall_time=float(s.wall_time))
                for s in accumulator.Scalars(tag)
            )

    merged_scalars: dict[str, list[ScalarPoint]] = {}
    for tag, points in scalars_by_tag.items():
        points.sort(key=lambda point: (point.step, point.wall_time))
        deduped_by_step = {}
        for point in points:
            deduped_by_step[point.step] = point
        merged_scalars[tag] = [deduped_by_step[step] for step in sorted(deduped_by_step)]
    return merged_scalars


def _load_cached_scalars(cache_path: Path) -> dict[str, dict[str, list[ScalarPoint]]]:
    cache_path = cache_path.expanduser().resolve()
    with cache_path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported scalar cache version in: {cache_path}")

    result = {}
    for run in payload.get("runs", []):
        result[str(run["label"])] = {
            str(tag): [ScalarPoint(*point) for point in points]
            for tag, points in run.get("scalars", {}).items()
        }
    if not result:
        raise ValueError(f"No run scalars found in cache: {cache_path}")
    print(f"[training curves] Reusing shared scalar cache: {cache_path}", flush=True)
    return result


def _available_tags(run_scalars_by_run: dict[str, dict[str, list[ScalarPoint]]]) -> list[str]:
    tags = set()
    for run_scalars in run_scalars_by_run.values():
        tags.update(run_scalars.keys())
    return sorted(tags)


def _select_default_tags(run_scalars_by_run: dict[str, dict[str, list[ScalarPoint]]]) -> list[str]:
    selected = [
        tag
        for tag in DEFAULT_TAG_PRIORITY
        if any(tag in run_scalars for run_scalars in run_scalars_by_run.values())
    ]
    if selected:
        return selected
    return _available_tags(run_scalars_by_run)[:3]


def _apply_step_limits(
    points: Iterable[ScalarPoint],
    min_step: int | None,
    max_step: int | None,
) -> list[ScalarPoint]:
    filtered = []
    for point in points:
        if min_step is not None and point.step < min_step:
            continue
        if max_step is not None and point.step > max_step:
            continue
        filtered.append(point)
    return filtered


def _ema(values: np.ndarray, smoothing: float) -> np.ndarray:
    if len(values) == 0 or smoothing <= 0.0:
        return values
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = smoothing * smoothed[index - 1] + (1.0 - smoothing) * values[index]
    return smoothed


def _fluctuation_band(
    raw_values: np.ndarray,
    smooth_values: np.ndarray,
    smoothing: float,
    band_smoothing: float | None = None,
) -> np.ndarray:
    deviation = np.abs(raw_values - smooth_values)
    effective_band_smoothing = (
        min(0.995, max(0.6, smoothing))
        if band_smoothing is None
        else float(band_smoothing)
    )
    return _ema(deviation, effective_band_smoothing)


def _downsample_indices(length: int, stride: int) -> np.ndarray:
    if stride <= 1 or length <= 2:
        return np.arange(length, dtype=int)
    indices = np.arange(0, length, stride, dtype=int)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return indices


def _points_to_xy(points: list[ScalarPoint], x_axis: str) -> tuple[np.ndarray, np.ndarray]:
    steps = np.asarray([point.step for point in points], dtype=float)
    values = np.asarray([point.value for point in points], dtype=float)

    if x_axis == "step":
        x_values = steps
    elif x_axis == "iteration":
        x_values = steps / float(TRAIN_NUM_ENVS)
    elif x_axis == "time":
        wall_time = np.asarray([point.wall_time for point in points], dtype=float)
        x_values = (wall_time - wall_time[0]) / 3600.0
    elif x_axis == "index":
        x_values = np.arange(len(points), dtype=float)
    else:
        raise ValueError(f"Unsupported x-axis: {x_axis}")

    return x_values, values


def _tensorboard_like_ylim(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return -0.05, 1.05
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if np.isclose(y_min, y_max):
        pad = 0.05 if np.isclose(y_min, 0.0) else abs(y_min) * 0.1
        return y_min - pad, y_max + pad
    span = y_max - y_min
    pad = span * 0.08
    return y_min - pad, y_max + pad


def _axis_label(x_axis: str) -> str:
    if x_axis == "step":
        return "Step"
    if x_axis == "iteration":
        return "Training Iteration (1e4)"
    if x_axis == "time":
        return "Wall-Clock Time (hours)"
    return "Point Index"


def _iteration_tick_formatter(value: float, _pos: int) -> str:
    scaled = value / 1e4
    if np.isclose(scaled, round(scaled)):
        return str(int(round(scaled)))
    return f"{scaled:.1f}"


def _sanitize_tag_for_filename(tag: str) -> str:
    if tag in DEFAULT_TAG_FILENAMES:
        return DEFAULT_TAG_FILENAMES[tag]
    return tag.replace("/", "_").replace(" ", "_")


def _display_title_for_tag(tag: str, fallback_title: str | None) -> str:
    if tag in DEFAULT_TAG_TITLES:
        return DEFAULT_TAG_TITLES[tag]
    return fallback_title or tag


def _tag_plot_setting(tag: str, key: str, default):
    return TAG_PLOT_OVERRIDES.get(tag, {}).get(key, default)


def _plot_tag(
    run_scalars_by_run: dict[str, dict[str, list[ScalarPoint]]],
    tag: str,
    *,
    run_colors: dict[str, str],
    x_axis: str,
    smoothing: float,
    downsample: int,
    show_raw: bool,
    min_step: int | None,
    max_step: int | None,
    title: str | None,
    figsize: tuple[float, float],
    linewidth: float,
):
    fig, axis = plt.subplots(figsize=figsize)
    plotted_any = False
    effective_smoothing = float(_tag_plot_setting(tag, "smoothing", smoothing))
    effective_downsample = int(_tag_plot_setting(tag, "downsample", downsample))
    legend_loc = str(_tag_plot_setting(tag, "legend_loc", "upper right"))
    band_scale = float(_tag_plot_setting(tag, "band_scale", 1.0))
    band_alpha = float(_tag_plot_setting(tag, "band_alpha", 0.14))
    band_smoothing = _tag_plot_setting(tag, "band_smoothing", None)
    all_band_values: list[np.ndarray] = []
    color_map = plt.get_cmap("tab10")
    for index, (run_label, run_scalars) in enumerate(run_scalars_by_run.items()):
        if tag not in run_scalars:
            continue
        points = _apply_step_limits(run_scalars[tag], min_step=min_step, max_step=max_step)
        if not points:
            continue

        x_values, raw_values = _points_to_xy(points, x_axis=x_axis)
        initial_indices = _downsample_indices(len(raw_values), effective_downsample)
        x_plot = x_values[initial_indices]
        raw_plot = raw_values[initial_indices]
        smooth_plot = _ema(raw_plot, effective_smoothing)
        band_plot = _fluctuation_band(
            raw_plot,
            smooth_plot,
            effective_smoothing,
            band_smoothing=band_smoothing,
        ) * band_scale
        color = run_colors.get(run_label, color_map(index % 10))

        axis.fill_between(
            x_plot,
            smooth_plot - band_plot,
            smooth_plot + band_plot,
            color=color,
            alpha=band_alpha,
            linewidth=0.0,
        )
        axis.plot(
            x_plot,
            smooth_plot,
            color=color,
            alpha=0.95,
            linewidth=linewidth,
            label=run_label,
        )
        all_band_values.append(smooth_plot - band_plot)
        all_band_values.append(smooth_plot + band_plot)
        plotted_any = True

    if all_band_values:
        axis.set_ylim(*_tensorboard_like_ylim(np.concatenate(all_band_values)))

    axis.set_xlabel(_axis_label(x_axis), fontsize=LABEL_FONTSIZE)
    axis.set_ylabel("Value", fontsize=LABEL_FONTSIZE)
    axis.grid(True, color="#d9d9d9", alpha=0.8, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_facecolor("white")
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.margins(x=0.01)
    axis.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if x_axis == "iteration":
        axis.xaxis.set_major_formatter(FuncFormatter(_iteration_tick_formatter))
    if not plotted_any:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
    fig.suptitle(_display_title_for_tag(tag, title), fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not (0.0 <= args.smoothing < 1.0):
        raise ValueError("--smoothing must be in the range [0, 1).")
    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1.")

    if args.scalar_cache is not None:
        run_scalars_by_run = _load_cached_scalars(args.scalar_cache)
    else:
        run_specs = _load_run_specs(args.config)
        runs = _resolve_runs(run_specs)
        print(
            f"[training curves] Reading {len(runs)} runs from {args.config.resolve()}",
            flush=True,
        )
        run_scalars_by_run = {run.label: _load_run_scalars(run) for run in runs}
    run_colors = _load_run_colors(args.config)

    tags = args.tags or PLOT_TAGS or _select_default_tags(run_scalars_by_run)
    if not tags:
        raise ValueError("No scalar tags were found in the resolved TensorBoard runs.")

    should_show = args.show
    output_path = args.output

    figures = []
    for tag in tags:
        fig = _plot_tag(
            run_scalars_by_run,
            tag,
            run_colors=run_colors,
            x_axis=args.x_axis,
            smoothing=args.smoothing,
            downsample=args.downsample,
            show_raw=args.show_raw,
            min_step=args.min_step,
            max_step=args.max_step,
            title=args.title,
            figsize=tuple(args.figsize),
            linewidth=args.linewidth,
        )
        figures.append((tag, fig))

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for tag, fig in figures:
        tagged_output = output_path.with_name(
            f"{output_path.stem}_{_sanitize_tag_for_filename(tag)}{output_path.suffix or '.png'}"
        )
        fig.savefig(tagged_output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to: {tagged_output}")

    if should_show:
        plt.show()
    for _, fig in figures:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot per-object unsafe metrics for TensorBoard runs listed in YAML."""

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

from ablation_raw_data import (
    OBJECT_PREPROCESSING,
    arrays_to_tuples,
    is_per_object_tag,
    load_ablation_dataset,
    read_tensorboard_scalars,
)


UNSAFE_REASON_NAMES = (
    "object_out_of_bound",
    "hand_too_far",
    "harmful_collision",
    "palm_flipped",
)

METRIC_SPECS = (
    (
        "unsafe_episode_rate",
        "unsafe_episode_rate",
        "UnsafeRateAvg",
    ),
    (
        "object_out_of_bound",
        "unsafe_reason_prop/object_out_of_bound",
        "Object-out-of-bound",
    ),
    (
        "hand_too_far",
        "unsafe_reason_prop/hand_too_far",
        "Hand-too-far",
    ),
    (
        "harmful_collision",
        "unsafe_reason_prop/harmful_collision",
        "Harmful-collision",
    ),
    (
        "palm_flipped",
        "unsafe_reason_prop/palm_flipped",
        "Palm-flipped",
    ),
)

OBJECT_ALIASES = {
    "1m0lvpzs": "windcart",
    "2kp2e9k7": "boot",
    "2oiqpnts": "toolbox",
    "z73ltdbb": "cow",
}

X_AXIS = "iteration"
TRAIN_NUM_ENVS = 32
SMOOTHING = float(OBJECT_PREPROCESSING["smoothing"])
DOWNSAMPLE = int(OBJECT_PREPROCESSING["downsample"])
MIN_STEP = None
MAX_STEP = None
SHOW_FIGURES = False
FIGSIZE = (7.0, 7.0)
DPI = 180
LINEWIDTH = 2.6
AVG_BAND_ALPHA = 0.16
AVG_BAND_SCALE = float(OBJECT_PREPROCESSING["band_scale"])
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "plots"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
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
    smoothed_value: float | None = None
    band: float | None = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot per-object unsafe metric comparisons from TensorBoard runs listed in YAML.",
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
        "--raw-data-dir",
        type=Path,
        default=None,
        help="NumPy dataset produced by export_ablation_raw_data.py. Takes precedence over --scalar-cache.",
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
        help="EMA smoothing factor in [0, 1).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=DOWNSAMPLE,
        help="Keep every Nth point before smoothing and plotting.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=MIN_STEP,
        help="Drop points before this TensorBoard step.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=MAX_STEP,
        help="Drop points after this TensorBoard step.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Root output directory for saved figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=SHOW_FIGURES,
        help="Open matplotlib windows after saving figures.",
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
    return parser


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
    raw_scalars = read_tensorboard_scalars(
        run.label,
        run.source_path,
        all_scalars=True,
    )
    return {
        tag: [ScalarPoint(*point) for point in points]
        for tag, points in arrays_to_tuples(raw_scalars).items()
    }


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
    print(f"[failure reasons] Reusing shared scalar cache: {cache_path}", flush=True)
    return result


def _load_raw_data_scalars(raw_data_dir: Path) -> dict[str, dict[str, list[ScalarPoint]]]:
    result = load_ablation_dataset(
        raw_data_dir,
        tag_filter=is_per_object_tag,
        point_factory=ScalarPoint,
    )
    print(
        f"[failure reasons] Loading compact NumPy plotting data: {raw_data_dir.resolve()}",
        flush=True,
    )
    return result


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


def _fluctuation_band(values: np.ndarray, smooth_values: np.ndarray, smoothing: float) -> np.ndarray:
    deviation = np.abs(values - smooth_values)
    return _ema(deviation, min(0.995, max(0.6, smoothing)))


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


def _axis_label(x_axis: str) -> str:
    if x_axis == "step":
        return "Step"
    if x_axis == "iteration":
        return "Training Iteration (1e4)"
    if x_axis == "time":
        return "Wall-Clock Time (hours)"
    return "Point Index"


def _sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _display_object_name(object_name: str) -> str:
    return OBJECT_ALIASES.get(object_name, object_name)


def _iteration_tick_formatter(value: float, _pos: int) -> str:
    scaled = value / 1e4
    if np.isclose(scaled, round(scaled)):
        return str(int(round(scaled)))
    return f"{scaled:.1f}"


def _tensorboard_like_ylim(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return -0.05, 1.05
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if np.isclose(y_min, y_max):
        pad = 0.05 if np.isclose(y_min, 0.0) else abs(y_min) * 0.1
        return y_min - pad, y_max + pad
    pad = (y_max - y_min) * 0.08
    return y_min - pad, y_max + pad


def _extract_object_series(
    run_scalars: dict[str, list[ScalarPoint]],
    metric_suffix: str,
) -> dict[str, list[ScalarPoint]]:
    object_series: dict[str, list[ScalarPoint]] = {}
    prefix = "train/"
    suffix = f"/{metric_suffix}"
    for tag, points in run_scalars.items():
        if not tag.startswith(prefix) or not tag.endswith(suffix):
            continue
        middle = tag[len(prefix) : -len(suffix)]
        if middle == "avg":
            continue
        object_series[middle] = points
    return object_series


def _discover_object_names(run_scalars_by_run: dict[str, dict[str, list[ScalarPoint]]]) -> list[str]:
    object_names = set()
    for run_scalars in run_scalars_by_run.values():
        object_names.update(_extract_object_series(run_scalars, "unsafe_episode_rate").keys())
    return sorted(object_names)


def _series_for_object(
    run_scalars: dict[str, list[ScalarPoint]],
    object_name: str,
    metric_suffix: str,
) -> list[ScalarPoint] | None:
    return run_scalars.get(f"train/{object_name}/{metric_suffix}")


def _plot_metric(
    run_scalars_by_run: dict[str, dict[str, list[ScalarPoint]]],
    *,
    run_colors: dict[str, str],
    object_name: str,
    metric_key: str,
    metric_suffix: str,
    title: str,
    x_axis: str,
    smoothing: float,
    downsample: int,
    min_step: int | None,
    max_step: int | None,
    figsize: tuple[float, float],
):
    fig, axis = plt.subplots(figsize=figsize)
    color_map = plt.get_cmap("tab10")
    plotted_any = False
    all_band_values: list[np.ndarray] = []

    for run_index, (run_label, run_scalars) in enumerate(run_scalars_by_run.items()):
        points = _series_for_object(run_scalars, object_name, metric_suffix)
        if not points:
            continue

        color = run_colors.get(run_label, color_map(run_index % 10))
        filtered_points = _apply_step_limits(points, min_step=min_step, max_step=max_step)
        if not filtered_points:
            continue

        avg_x, avg_values = _points_to_xy(filtered_points, x_axis=x_axis)
        if (
            filtered_points[0].smoothed_value is not None
            and filtered_points[0].band is not None
        ):
            # NumPy export already applied per-object smoothing and downsampling.
            avg_x_plot = avg_x
            avg_smooth_plot = np.asarray(
                [point.smoothed_value for point in filtered_points],
                dtype=float,
            )
            avg_band_plot = np.asarray(
                [point.band for point in filtered_points],
                dtype=float,
            )
        else:
            avg_indices = _downsample_indices(len(avg_values), downsample)
            avg_x_plot = avg_x[avg_indices]
            avg_value_plot = avg_values[avg_indices]
            avg_smooth_plot = _ema(avg_value_plot, smoothing)
            avg_band_plot = (
                _fluctuation_band(avg_value_plot, avg_smooth_plot, smoothing)
                * AVG_BAND_SCALE
            )

        axis.fill_between(
            avg_x_plot,
            avg_smooth_plot - avg_band_plot,
            avg_smooth_plot + avg_band_plot,
            color=color,
            alpha=AVG_BAND_ALPHA,
            linewidth=0.0,
        )
        axis.plot(
            avg_x_plot,
            avg_smooth_plot,
            color=color,
            alpha=0.95,
            linewidth=LINEWIDTH,
            label=run_label,
        )

        all_band_values.append(avg_smooth_plot - avg_band_plot)
        all_band_values.append(avg_smooth_plot + avg_band_plot)
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
    else:
        axis.legend(loc="upper right", fontsize=11, frameon=True)

    fig.suptitle(f"{_display_object_name(object_name)}: {title}", fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def main() -> int:
    args = _build_parser().parse_args()
    if not (0.0 <= args.smoothing < 1.0):
        raise ValueError("--smoothing must be in the range [0, 1).")
    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1.")

    if args.raw_data_dir is not None:
        run_scalars_by_run = _load_raw_data_scalars(args.raw_data_dir)
    elif args.scalar_cache is not None:
        run_scalars_by_run = _load_cached_scalars(args.scalar_cache)
    else:
        runs = _resolve_runs(_load_run_specs(args.config))
        print(
            f"[failure reasons] Reading {len(runs)} runs from {args.config.resolve()}",
            flush=True,
        )
        run_scalars_by_run = {run.label: _load_run_scalars(run) for run in runs}
    run_colors = _load_run_colors(args.config)
    object_names = _discover_object_names(run_scalars_by_run)
    if not object_names:
        raise ValueError("No per-object unsafe tags were found in the resolved TensorBoard runs.")

    figures = []
    for object_name in object_names:
        for metric_key, metric_suffix, title in METRIC_SPECS:
            fig = _plot_metric(
                run_scalars_by_run,
                run_colors=run_colors,
                object_name=object_name,
                metric_key=metric_key,
                metric_suffix=metric_suffix,
                title=title,
                x_axis=args.x_axis,
                smoothing=args.smoothing,
                downsample=args.downsample,
                min_step=args.min_step,
                max_step=args.max_step,
                figsize=tuple(args.figsize),
            )
            figures.append((object_name, metric_key, fig))

    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for object_name, metric_key, fig in figures:
        object_dir = output_root / _sanitize_name(object_name)
        object_dir.mkdir(parents=True, exist_ok=True)
        tagged_output = object_dir / f"{_sanitize_name(metric_key)}.png"
        fig.savefig(tagged_output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to: {tagged_output}")

    if args.show:
        plt.show()
    for _, _, fig in figures:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

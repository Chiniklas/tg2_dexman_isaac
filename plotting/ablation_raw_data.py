#!/usr/bin/env python3
"""TensorBoard extraction and compact NumPy I/O for ablation plots."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import numpy as np
import yaml


DATASET_VERSION = 3
MANIFEST_NAME = "manifest.json"

RAW_SCALAR_DTYPE = np.dtype(
    [
        ("step", np.int64),
        ("value", np.float64),
        ("wall_time", np.float64),
    ]
)
PLOTTING_SCALAR_DTYPE = np.dtype(
    [
        ("step", np.int64),
        ("value", np.float64),
        ("wall_time", np.float64),
        ("smoothed_value", np.float64),
        ("band", np.float64),
    ]
)

BASE_TAGS = {
    "beta",
    "train/avg/unsafe_episode_rate",
    "eval/avg/lift_success",
}
PER_OBJECT_SUFFIXES = (
    "/unsafe_episode_rate",
    "/unsafe_reason_prop/object_out_of_bound",
    "/unsafe_reason_prop/hand_too_far",
    "/unsafe_reason_prop/harmful_collision",
    "/unsafe_reason_prop/palm_flipped",
)
TRAINING_PREPROCESSING = {
    "beta": {
        "smoothing": 0.999,
        "rolling_window": 501,
        "downsample": 100,
        "band_smoothing": 0.999,
        "band_scale": 1.0,
    },
    "train/avg/unsafe_episode_rate": {
        "smoothing": 0.9995,
        "rolling_window": 1201,
        "downsample": 200,
        "band_smoothing": 0.999,
        "band_scale": 1.0,
    },
    "eval/avg/lift_success": {
        "smoothing": 0.75,
        "rolling_window": None,
        "downsample": 1,
        "band_smoothing": None,
        "band_scale": 1.0,
    },
}
OBJECT_PREPROCESSING = {
    "smoothing": 0.97,
    "rolling_window": None,
    "downsample": 10,
    "band_smoothing": None,
    "band_scale": 1.0,
}


def import_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard event reading requires the 'tensorboard' package."
        ) from exc
    return event_accumulator


def load_enabled_runs(config_path: Path) -> list[tuple[str, Path]]:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Plot config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    raw_runs = config.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"Expected a non-empty 'runs' list in: {config_path}")

    runs = []
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
        runs.append((label, run_path.resolve()))

    if not runs:
        raise ValueError(f"No enabled runs were found in: {config_path}")
    return runs


def find_event_files(run_path: Path) -> list[Path]:
    if not run_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {run_path}")
    if run_path.is_file():
        return [run_path]
    event_files = sorted(
        path for path in run_path.rglob("events.out.tfevents*") if path.is_file()
    )
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {run_path}")
    return event_files


def is_per_object_tag(tag: str) -> bool:
    if not tag.startswith("train/"):
        return False
    object_name = tag[len("train/") :].split("/", 1)[0]
    return object_name != "avg" and any(
        tag.endswith(suffix) for suffix in PER_OBJECT_SUFFIXES
    )


def wanted_ablation_tag(tag: str) -> bool:
    return tag in BASE_TAGS or is_per_object_tag(tag)


def read_tensorboard_scalars(
    label: str,
    run_path: Path,
    *,
    all_scalars: bool = False,
) -> dict[str, np.ndarray]:
    """Read and deduplicate TensorBoard scalars for one configured run."""
    event_accumulator = import_event_accumulator()
    size_guidance = {event_accumulator.SCALARS: 0}
    scalars_by_tag: dict[str, list[tuple[int, float, float]]] = defaultdict(list)

    for event_file in find_event_files(run_path):
        size_mb = event_file.stat().st_size / (1024.0 * 1024.0)
        print(
            f"[plot data] Loading {label}: {event_file.name} ({size_mb:.1f} MB)...",
            flush=True,
        )
        accumulator = event_accumulator.EventAccumulator(
            str(event_file), size_guidance=size_guidance
        )
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        selected_tags = (
            list(tags)
            if all_scalars
            else [tag for tag in tags if wanted_ablation_tag(tag)]
        )
        for tag in selected_tags:
            scalars_by_tag[tag].extend(
                (int(point.step), float(point.value), float(point.wall_time))
                for point in accumulator.Scalars(tag)
            )
        print(
            f"[plot data] Loaded {label}: retained {len(selected_tags)} scalar tags.",
            flush=True,
        )

    merged = {}
    for tag, points in scalars_by_tag.items():
        points.sort(key=lambda point: (point[0], point[2]))
        deduped = {point[0]: point for point in points}
        ordered = [deduped[step] for step in sorted(deduped)]
        merged[tag] = np.asarray(ordered, dtype=RAW_SCALAR_DTYPE)
    return merged


def _ema(values: np.ndarray, smoothing: float) -> np.ndarray:
    if len(values) == 0 or smoothing <= 0.0:
        return values.astype(float, copy=True)
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = (
            smoothing * smoothed[index - 1] + (1.0 - smoothing) * values[index]
        )
    return smoothed


def _centered_moving_average(values: np.ndarray, window: int | None) -> np.ndarray:
    if window is None or window <= 1 or len(values) <= 2:
        return values
    effective_window = min(int(window), len(values))
    if effective_window % 2 == 0:
        effective_window -= 1
    if effective_window <= 1:
        return values
    pad = effective_window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(effective_window, dtype=float) / float(effective_window)
    return np.convolve(padded, kernel, mode="valid")


def _fluctuation_band(
    raw_values: np.ndarray,
    smooth_values: np.ndarray,
    smoothing: float,
    band_smoothing: float | None,
) -> np.ndarray:
    deviation = np.abs(raw_values - smooth_values)
    effective_smoothing = (
        min(0.995, max(0.6, smoothing))
        if band_smoothing is None
        else float(band_smoothing)
    )
    return _ema(deviation, effective_smoothing)


def _downsample_indices(length: int, stride: int) -> np.ndarray:
    if stride <= 1 or length <= 2:
        return np.arange(length, dtype=int)
    indices = np.arange(0, length, stride, dtype=int)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return indices


def preprocessing_for_tag(tag: str) -> tuple[str, dict[str, Any]]:
    if tag in TRAINING_PREPROCESSING:
        return "training_curve", dict(TRAINING_PREPROCESSING[tag])
    if is_per_object_tag(tag):
        return "per_object_curve", dict(OBJECT_PREPROCESSING)
    raise ValueError(f"No plotting preprocessing is configured for tag: {tag}")


def preprocess_for_plotting(tag: str, raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the plot's smoothing/downsampling before writing the compact array."""
    kind, settings = preprocessing_for_tag(tag)
    raw_values = np.asarray(raw["value"], dtype=float)
    smoothing = float(settings["smoothing"])
    downsample = int(settings["downsample"])

    if kind == "per_object_curve":
        indices = _downsample_indices(len(raw_values), downsample)
        selected_raw = raw_values[indices]
        smoothed = _ema(selected_raw, smoothing)
        band = _fluctuation_band(
            selected_raw,
            smoothed,
            smoothing,
            settings["band_smoothing"],
        ) * float(settings["band_scale"])
    else:
        smoothed_full = _ema(raw_values, smoothing)
        smoothed_full = _centered_moving_average(
            smoothed_full,
            settings["rolling_window"],
        )
        band_full = _fluctuation_band(
            raw_values,
            smoothed_full,
            smoothing,
            settings["band_smoothing"],
        ) * float(settings["band_scale"])
        indices = _downsample_indices(len(raw_values), downsample)
        selected_raw = raw_values[indices]
        smoothed = smoothed_full[indices]
        band = band_full[indices]

    compact = np.empty(len(indices), dtype=PLOTTING_SCALAR_DTYPE)
    compact["step"] = raw["step"][indices]
    compact["value"] = selected_raw
    compact["wall_time"] = raw["wall_time"][indices]
    compact["smoothed_value"] = smoothed
    compact["band"] = band
    metadata = {"kind": kind, **settings}
    return compact, metadata


def _encoded_filename(value: str) -> str:
    return f"{quote(value, safe='')}.npy"


def export_ablation_dataset(
    *,
    config_path: Path,
    output_dir: Path,
) -> Path:
    """Write compact arrays into one timestamped plotting output."""
    config_path = config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    runs = load_enabled_runs(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_runs = []
    for label, run_path in runs:
        data_dir = output_dir / run_path.name
        data_dir.mkdir(parents=True, exist_ok=True)
        raw_scalars = read_tensorboard_scalars(label, run_path)
        tag_files = {}
        preprocessing = {}
        compact_count = 0
        for tag, raw_values in sorted(raw_scalars.items()):
            compact, settings = preprocess_for_plotting(tag, raw_values)
            data_path = data_dir / _encoded_filename(tag)
            np.save(data_path, compact, allow_pickle=False)
            tag_files[tag] = data_path.name
            preprocessing[tag] = settings
            compact_count += len(compact)

        run_manifest = {
            "version": DATASET_VERSION,
            "label": label,
            "source_path": str(run_path),
            "data_dir": str(data_dir),
            "format": {
                "container": "npy",
                "dtype": PLOTTING_SCALAR_DTYPE.descr,
                "columns": list(PLOTTING_SCALAR_DTYPE.names or ()),
            },
            "tags": tag_files,
            "preprocessing": preprocessing,
        }
        (data_dir / MANIFEST_NAME).write_text(
            json.dumps(run_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_runs.append(run_manifest)
        raw_count = sum(len(values) for values in raw_scalars.values())
        print(
            f"[plot data] Saved {label} in {data_dir}: "
            f"{raw_count:,} -> {compact_count:,} points.",
            flush=True,
        )

    manifest = {
        "version": DATASET_VERSION,
        "config_path": str(config_path),
        "storage": "inside_plotting_output",
        "runs": manifest_runs,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[plot data] Dataset index ready: {manifest_path}", flush=True)
    return manifest_path


def load_ablation_dataset(
    dataset_dir: Path,
    *,
    tag_filter: Callable[[str], bool] | None = None,
    point_factory: Callable[[int, float, float, float, float], Any] | None = None,
) -> dict[str, dict[str, list[Any]]]:
    """Load compact per-run arrays without importing TensorBoard."""
    dataset_dir = dataset_dir.expanduser().resolve()
    manifest_path = dataset_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Plot-data manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != DATASET_VERSION:
        raise ValueError(
            f"Unsupported plot-data version in {manifest_path}: "
            f"found {manifest.get('version')}, expected {DATASET_VERSION}. "
            "Regenerate this timestamp's raw_data directory."
        )

    result = {}
    for run in manifest.get("runs", []):
        label = str(run["label"])
        data_dir = Path(run["data_dir"]).expanduser().resolve()
        run_scalars = {}
        for tag, filename in run.get("tags", {}).items():
            tag = str(tag)
            if tag_filter is not None and not tag_filter(tag):
                continue
            data_path = (data_dir / filename).resolve()
            if not data_path.is_relative_to(data_dir):
                raise ValueError(f"Plot-data path escapes run directory: {filename}")
            values = np.load(data_path, allow_pickle=False)
            if values.dtype.names != PLOTTING_SCALAR_DTYPE.names:
                raise ValueError(f"Unexpected plotting scalar dtype in: {data_path}")
            rows = [
                (
                    int(row["step"]),
                    float(row["value"]),
                    float(row["wall_time"]),
                    float(row["smoothed_value"]),
                    float(row["band"]),
                )
                for row in values
            ]
            run_scalars[tag] = (
                rows
                if point_factory is None
                else [point_factory(*row) for row in rows]
            )
        result[label] = run_scalars

    if not result:
        raise ValueError(f"No ablation runs found in: {manifest_path}")
    return result


def arrays_to_tuples(
    scalars: dict[str, np.ndarray],
) -> dict[str, list[tuple[int, float, float]]]:
    """Convert unprocessed arrays for the backward-compatible direct-read path."""
    return {
        tag: [
            (int(row["step"]), float(row["value"]), float(row["wall_time"]))
            for row in values
        ]
        for tag, values in scalars.items()
    }

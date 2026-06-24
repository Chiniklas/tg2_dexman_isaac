#!/usr/bin/env python3
"""Read TensorBoard runs once and cache the scalars used by all paper plots."""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import yaml


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache the TensorBoard scalars required by the plotting scripts."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _import_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard event reading requires the 'tensorboard' package."
        ) from exc
    return event_accumulator


def _load_enabled_runs(config_path: Path) -> list[tuple[str, Path]]:
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    raw_runs = config.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"Expected a non-empty 'runs' list in: {config_path}")

    runs = []
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise ValueError(f"Run entry {index} must be a mapping.")
        if not raw_run.get("enabled", True):
            continue
        label = str(raw_run.get("label", "")).strip()
        raw_path = str(raw_run.get("path", "")).strip()
        if not label or not raw_path:
            raise ValueError(
                f"Enabled run entry {index} requires non-empty 'label' and 'path'."
            )
        run_path = Path(raw_path).expanduser()
        if not run_path.is_absolute():
            run_path = config_path.parent / run_path
        runs.append((label, run_path.resolve()))

    if not runs:
        raise ValueError(f"No enabled runs were found in: {config_path}")
    return runs


def _find_event_files(run_path: Path) -> list[Path]:
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


def _wanted_tag(tag: str) -> bool:
    if tag in BASE_TAGS:
        return True
    return tag.startswith("train/") and any(
        tag.endswith(suffix) for suffix in PER_OBJECT_SUFFIXES
    )


def _read_run(label: str, run_path: Path) -> dict[str, list[tuple[int, float, float]]]:
    event_accumulator = _import_event_accumulator()
    size_guidance = {event_accumulator.SCALARS: 0}
    scalars_by_tag: dict[str, list[tuple[int, float, float]]] = defaultdict(list)

    for event_file in _find_event_files(run_path):
        size_mb = event_file.stat().st_size / (1024.0 * 1024.0)
        print(
            f"[shared load] Loading {label}: {event_file.name} ({size_mb:.1f} MB)...",
            flush=True,
        )
        accumulator = event_accumulator.EventAccumulator(
            str(event_file), size_guidance=size_guidance
        )
        accumulator.Reload()
        wanted_tags = [
            tag for tag in accumulator.Tags().get("scalars", []) if _wanted_tag(tag)
        ]
        for tag in wanted_tags:
            scalars_by_tag[tag].extend(
                (int(point.step), float(point.value), float(point.wall_time))
                for point in accumulator.Scalars(tag)
            )
        print(
            f"[shared load] Loaded {label}: retained {len(wanted_tags)} scalar tags.",
            flush=True,
        )

    merged = {}
    for tag, points in scalars_by_tag.items():
        points.sort(key=lambda point: (point[0], point[2]))
        deduped = {point[0]: point for point in points}
        merged[tag] = [deduped[step] for step in sorted(deduped)]
    return merged


def main() -> int:
    args = _build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    runs = _load_enabled_runs(config_path)

    print(f"[shared load] Reading {len(runs)} TensorBoard runs once.", flush=True)
    payload = {
        "version": 1,
        "config_path": str(config_path),
        "runs": [
            {
                "label": label,
                "source_path": str(run_path),
                "scalars": _read_run(label, run_path),
            }
            for label, run_path in runs
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[shared load] Scalar cache ready: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

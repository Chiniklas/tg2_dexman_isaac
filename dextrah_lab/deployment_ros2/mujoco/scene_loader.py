from __future__ import annotations

import argparse
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML_PATH = REPO_ROOT / "dextrah_lab" / "assets" / "tiangong2pro" / "xml" / "tiangong2.0_pro_with_hands_half_body.xml"


def _require_mujoco():
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo is required for the scene loader. Install it with `python -m pip install -e .`."
        ) from exc
    return mujoco


def _maybe_import_viewer():
    try:
        import mujoco.viewer  # type: ignore
    except ModuleNotFoundError:
        return None
    return mujoco.viewer


def load_model(xml_path: Path):
    mujoco = _require_mujoco()
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return mujoco, model, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and preview a MuJoCo scene XML.")
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML_PATH,
        help="Path to the MuJoCo XML scene to load.",
    )
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="Only validate the XML and print model stats without opening the viewer.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Optional simulation timestep override.",
    )
    args = parser.parse_args()

    xml_path = args.xml.expanduser().resolve()
    mujoco, model, data = load_model(xml_path)
    if args.dt is not None:
        model.opt.timestep = float(args.dt)

    print(f"Loaded MuJoCo XML: {xml_path}")
    print(f"  joints: {model.njnt}")
    print(f"  bodies: {model.nbody}")
    print(f"  geoms:  {model.ngeom}")
    print(f"  dt:     {model.opt.timestep}")

    if args.headless_check:
        return

    viewer_mod = _maybe_import_viewer()
    if viewer_mod is None:
        raise ModuleNotFoundError("mujoco.viewer is required to open the interactive scene viewer.")

    with viewer_mod.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

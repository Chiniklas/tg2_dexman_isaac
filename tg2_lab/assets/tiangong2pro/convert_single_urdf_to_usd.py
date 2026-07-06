# Copyright (c) 2026
#
# Utility to convert one URDF robot asset into USD with Isaac Lab's URDF
# converter. It can either keep control gains out of the USD for IsaacLab
# runtime actuator cfgs, or author conservative gains for Physics Inspector use.

from __future__ import annotations

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Convert a single URDF robot asset to USD.")
parser.add_argument("input", type=Path, help="Path to the input URDF file.")
parser.add_argument("output_dir", type=Path, help="Directory where the USD asset folder/file should be written.")
parser.add_argument("--usd-name", type=str, default=None, help="Generated USD file name. Defaults to input stem.")
parser.add_argument("--fix-base", action="store_true", help="Fix the root link during URDF import.")
parser.add_argument(
    "--merge-fixed-joints",
    action="store_true",
    help="Merge fixed joints during import. Leave disabled to preserve link names and contact bodies.",
)
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=True,
    help="Make the generated asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--drive-profile",
    choices=("zero", "inspector", "sharpa_contact", "importer"),
    default="zero",
    help=(
        "Post-process imported angular drives. 'zero' clears gains for IsaacLab runtime actuators; "
        "'inspector' writes stiff Kp/Kd for Physics Inspector control; "
        "'sharpa_contact' writes KUKA+SHARPA-inspired gains/friction/armature for contact-rich RL; "
        "'importer' leaves Isaac defaults."
    ),
)
parser.add_argument(
    "--mimic-natural-frequency",
    type=float,
    default=0.0,
    help=(
        "Natural frequency for PhysX mimic joint compliance in Hz. "
        "Use <= 0 for a hard mimic constraint."
    ),
)
parser.add_argument(
    "--mimic-damping-ratio",
    type=float,
    default=0.0,
    help=(
        "Damping ratio for PhysX mimic joint compliance. "
        "Use <= 0 with non-positive natural frequency for a hard mimic constraint."
    ),
)
parser.add_argument(
    "--fully-actuated",
    action="store_true",
    help="Strip URDF mimic tags before import so follower finger joints become normal controllable DOFs.",
)
parser.add_argument(
    "--self-collision",
    action="store_true",
    help="Enable self-collision in the imported articulation USD.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from pxr import Sdf, Usd, UsdPhysics  # noqa: E402


def _prepare_import_urdf(source: Path, fully_actuated: bool) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    """Create an importer-local URDF copy with mesh paths normalized."""
    temp_dir = tempfile.TemporaryDirectory(prefix="tiangong2pro_urdf_import_")
    temp_root = Path(temp_dir.name)
    temp_urdf_dir = temp_root / "urdf"
    temp_urdf_dir.mkdir(parents=True, exist_ok=True)

    source = source.resolve()
    temp_source = temp_urdf_dir / source.name
    text = source.read_text()
    text = text.replace('filename="./meshes/', 'filename="../meshes/')
    if fully_actuated:
        root = ET.fromstring(text)
        removed = 0
        for joint in root.findall("joint"):
            mimic = joint.find("mimic")
            if mimic is not None:
                joint.remove(mimic)
                removed += 1
        ET.ElementTree(root).write(temp_source, encoding="unicode", xml_declaration=False)
        print(f"Stripped {removed} URDF mimic tags for fully actuated import.")
    else:
        temp_source.write_text(text)

    source_mesh_dir = source.parent.parent / "meshes"
    temp_mesh_dir = temp_root / "meshes"
    if not source_mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {source_mesh_dir}")
    temp_mesh_dir.symlink_to(source_mesh_dir.resolve(), target_is_directory=True)
    return temp_dir, temp_source


def _mimic_specs(urdf_path: Path) -> dict[str, dict[str, float | str]]:
    root = ET.parse(urdf_path).getroot()
    specs = {}
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if "name" not in joint.attrib or mimic is None:
            continue
        specs[joint.attrib["name"]] = {
            "joint": mimic.attrib["joint"],
            "multiplier": float(mimic.attrib.get("multiplier", "1.0")),
            "offset": float(mimic.attrib.get("offset", "0.0")),
        }
    return specs


def _inspector_gains(joint_name: str, mimic_joints: set[str]) -> tuple[float, float]:
    if joint_name in mimic_joints:
        return 0.0, 0.0
    if joint_name == "shoulder_pitch_r_joint":
        return 60.0, 3.0
    if joint_name == "shoulder_roll_r_joint":
        return 20.0, 1.5
    if joint_name.startswith("head_"):
        return 10.0, 1.0
    if any(token in joint_name for token in ("shoulder_", "elbow_", "wrist_")):
        return 10.0, 1.0
    if any(token in joint_name for token in ("index_", "middle_", "ring_", "little_", "thumb_")):
        return 100.0, 10.0
    return 0.0, 0.0


def _sharpa_contact_gains(joint_name: str, mimic_joints: set[str]) -> tuple[float, float]:
    """KUKA+SHARPA-inspired starting point for contact-rich training.

    The reference asset uses strong arm drives and comparatively compliant
    finger drives, with armature/friction providing extra damping on the hand.
    Mimic followers still get zero drives because the mimic constraint owns them.
    """
    if joint_name in mimic_joints:
        return 0.0, 0.0
    if joint_name == "shoulder_pitch_r_joint":
        return 600.0, 27.0
    if joint_name == "shoulder_roll_r_joint":
        return 600.0, 27.0
    if joint_name == "shoulder_yaw_r_joint":
        return 500.0, 24.7
    if joint_name == "elbow_pitch_r_joint":
        return 400.0, 22.1
    if joint_name in {"elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint"}:
        return 200.0, 9.2
    if joint_name.startswith("head_"):
        return 10.0, 1.0
    if joint_name in {"index_joint_0", "middle_joint_0", "ring_joint_0", "little_joint_0"}:
        return 4.76, 0.21
    if joint_name in {"index_joint_1", "middle_joint_1", "ring_joint_1", "little_joint_1"}:
        return 0.9, 0.042
    if joint_name in {"thumb_joint_0", "thumb_joint_1"}:
        return 6.95, 0.29
    if joint_name == "thumb_joint_2":
        return 4.76, 0.21
    if joint_name == "thumb_joint_3":
        return 0.9, 0.042
    return 0.0, 0.0


def _inspector_max_force(joint_name: str, mimic_joints: set[str]) -> float | None:
    if joint_name in mimic_joints:
        return 0.0
    if any(token in joint_name for token in ("index_", "middle_", "ring_", "little_", "thumb_")):
        return 100.0
    return None


def _sharpa_contact_max_force(joint_name: str, mimic_joints: set[str]) -> float | None:
    if joint_name in mimic_joints:
        return 0.0
    if joint_name in {"index_joint_0", "middle_joint_0", "ring_joint_0", "little_joint_0"}:
        return 1.864
    if joint_name in {"index_joint_1", "middle_joint_1", "ring_joint_1", "little_joint_1"}:
        return 0.638
    if joint_name in {"thumb_joint_0", "thumb_joint_1"}:
        return 3.3
    if joint_name == "thumb_joint_2":
        return 1.864
    if joint_name == "thumb_joint_3":
        return 0.638
    if any(token in joint_name for token in ("shoulder_", "elbow_", "wrist_")):
        return 300.0
    return None


def _sharpa_contact_passive_props(joint_name: str) -> tuple[float, float] | None:
    if joint_name in {"index_joint_0", "middle_joint_0", "ring_joint_0", "little_joint_0"}:
        return 0.00265, 0.07456
    if joint_name in {"index_joint_1", "middle_joint_1", "ring_joint_1", "little_joint_1"}:
        return 0.0006, 0.01276
    if joint_name in {"thumb_joint_0", "thumb_joint_1"}:
        return 0.0032, 0.132
    if joint_name == "thumb_joint_2":
        return 0.00265, 0.07456
    if joint_name == "thumb_joint_3":
        return 0.0006, 0.01276
    return None


def _postprocess_joint_drives(usd_path: str, urdf_path: Path, profile: str) -> int:
    """Adjust importer-authored joint drives for the selected usage profile."""
    if profile == "importer":
        return 0
    stage = Usd.Stage.Open(usd_path)
    mimic_specs = _mimic_specs(urdf_path)
    mimic_joints = set(mimic_specs)
    updated = 0
    for prim in stage.Traverse():
        if "PhysicsDriveAPI:angular" not in [str(api) for api in prim.GetAppliedSchemas()]:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if profile == "inspector":
            stiffness, damping = _inspector_gains(prim.GetName(), mimic_joints)
        elif profile == "sharpa_contact":
            stiffness, damping = _sharpa_contact_gains(prim.GetName(), mimic_joints)
        else:
            stiffness, damping = 0.0, 0.0
        drive.GetStiffnessAttr().Set(stiffness)
        drive.GetDampingAttr().Set(damping)
        if profile == "sharpa_contact":
            max_force = _sharpa_contact_max_force(prim.GetName(), mimic_joints)
            passive_props = _sharpa_contact_passive_props(prim.GetName())
            if passive_props is not None:
                armature, friction = passive_props
                prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(armature)
                prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(friction)
        else:
            max_force = _inspector_max_force(prim.GetName(), mimic_joints)
        if max_force is not None:
            drive.GetMaxForceAttr().Set(max_force)
        drive.GetTypeAttr().Set("force")
        updated += 1
    stage.GetRootLayer().Save()
    return updated


def _postprocess_mimic_compliance(usd_path: str, natural_frequency: float, damping_ratio: float) -> int:
    """Author PhysX mimic compliance settings on importer-created mimic APIs."""
    stage = Usd.Stage.Open(usd_path)
    updated = 0
    for prim in stage.Traverse():
        applied_schemas = {str(api) for api in prim.GetAppliedSchemas()}
        for axis in ("rotX", "rotY", "rotZ"):
            if f"PhysxMimicJointAPI:{axis}" not in applied_schemas:
                continue
            prim.CreateAttribute(
                f"physxMimicJoint:{axis}:naturalFrequency",
                Sdf.ValueTypeNames.Float,
            ).Set(float(natural_frequency))
            prim.CreateAttribute(
                f"physxMimicJoint:{axis}:dampingRatio",
                Sdf.ValueTypeNames.Float,
            ).Set(float(damping_ratio))
            updated += 1
    stage.GetRootLayer().Save()
    return updated


def main() -> None:
    source_urdf = args_cli.input.resolve()
    usd_root = args_cli.output_dir.resolve()
    usd_name = args_cli.usd_name or f"{source_urdf.stem}.usd"
    if not usd_name.endswith((".usd", ".usda")):
        usd_name = f"{usd_name}.usd"

    asset_dir = usd_root / source_urdf.stem
    asset_dir.mkdir(parents=True, exist_ok=True)

    temp_dir, import_urdf = _prepare_import_urdf(source_urdf, args_cli.fully_actuated)
    try:
        cfg = UrdfConverterCfg(
            asset_path=str(import_urdf),
            usd_dir=str(asset_dir),
            usd_file_name=usd_name,
            fix_base=args_cli.fix_base,
            merge_fixed_joints=args_cli.merge_fixed_joints,
            force_usd_conversion=True,
            make_instanceable=args_cli.make_instanceable,
            convert_mimic_joints_to_normal_joints=not args_cli.fully_actuated,
            joint_drive=None,
            collision_from_visuals=False,
            collider_type="convex_decomposition",
            self_collision=args_cli.self_collision,
            replace_cylinders_with_capsules=True,
        )

        print("-" * 80)
        print(f"Input URDF file: {source_urdf}")
        print(f"Temporary import URDF: {import_urdf}")
        print("URDF importer config:")
        print_dict(cfg.to_dict(), nesting=0)
        print("-" * 80)

        converter = UrdfConverter(cfg)
        updated_drives = _postprocess_joint_drives(converter.usd_path, import_urdf, args_cli.drive_profile)
        updated_mimics = _postprocess_mimic_compliance(
            converter.usd_path,
            args_cli.mimic_natural_frequency,
            args_cli.mimic_damping_ratio,
        )
        print(f"Post-processed angular drives with profile '{args_cli.drive_profile}' on {updated_drives} joints.")
        print(
            "Post-processed mimic compliance on "
            f"{updated_mimics} joints with naturalFrequency={args_cli.mimic_natural_frequency}, "
            f"dampingRatio={args_cli.mimic_damping_ratio}."
        )
        print("URDF importer output:")
        print(f"Generated USD file: {converter.usd_path}")
        print("-" * 80)
    finally:
        shutil.rmtree(temp_dir.name, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

#!/usr/bin/env bash
set -euo pipefail

# Install the package metadata first so the normal Isaac/Python pins are applied.
python -m pip install -e .

# urdfpy 0.0.22 works for this project, but its metadata pins networkx==2.2.
# That NetworkX version is incompatible with Python 3.11 and Isaac Sim 5, so
# install urdfpy without its stale transitive dependencies and restore Isaac's pins.
python -m pip install --no-deps urdfpy==0.0.22
python -m pip install \
  "numpy>=1.23.5,<2.0.0" \
  "networkx==3.3" \
  "pillow==11.3.0"

python - <<'PY'
import networkx
import numpy
import urdfpy
import warp as wp

from dextrah_lab.utils.import_urdf import parse_urdf_annotated  # noqa: F401

urdfpy.URDF.load("dextrah_lab/assets/tg2_inspirehand/urdf/tg2_with_hands_no_legs.urdf")

print(f"numpy {numpy.__version__}")
print(f"networkx {networkx.__version__}")
print("urdfpy ok")
print(f"warp {wp.__version__}")
print("dextrah_lab URDF importer ok")
print("TG2 URDF load ok")
PY

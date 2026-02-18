from __future__ import annotations

from typing import Any


class Kinematics:
    """Compatibility wrapper around the current kinematics backend.

    This local wrapper lets downstream code import from `dextrah_lab.utils.*`
    today while we phase out direct `fabrics_sim.*` imports.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        try:
            from fabrics_sim.prod.kinematics import Kinematics as _BackendKinematics
        except ImportError as exc:
            raise ImportError(
                "No local kinematics backend configured yet. "
                "Install/keep fabrics_sim for now, or implement a native backend in "
                "`dextrah_lab/utils/kinematics.py`."
            ) from exc

        self._backend = _BackendKinematics(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.eval(*args, **kwargs)

    def get_link_index(self, link_name: str) -> int:
        return int(self._backend.get_link_index(link_name))

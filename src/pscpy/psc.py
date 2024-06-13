from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .adios2py import File


class RunInfo:
    """Global information about the PSC run

    Currently stores domain info.
    TODO: Should also know about timestep, species, whatever...
    """

    def __init__(self, file: File, length: ArrayLike | None = None, corner: ArrayLike | None = None):
        assert len(file.variable_names) > 0
        var = next(iter(file.variable_names))
        self.gdims = np.asarray(file.get_variable(var).shape)[0:3]

        if length is not None:
            self.length = np.asarray(length)
        elif "length" in file.attribute_names:
            self.length = np.asarray(file.get_attribute("length"))
        else:
            self.length = self.gdims

        if corner is not None:
            self.corner = np.asarray(corner)
        elif "corner" in file.attribute_names:
            self.corner = np.asanyarray(file.get_attribute("corner"))
        elif "length" in file.attribute_names:
            self.corner = -0.5 * self.length
        else:
            self.corner = np.array([0.0, 0.0, 0.0])

        self.x = self._get_coord(0)
        self.y = self._get_coord(1)
        self.z = self._get_coord(2)

    def _get_coord(self, coord_idx: int) -> NDArray:
        return np.linspace(self.corner[coord_idx], self.corner[coord_idx] + self.length[coord_idx], self.gdims[coord_idx], endpoint=False)

    def __repr__(self):
        return f"Psc(gdims={self.gdims}, length={self.length}, corner={self.corner})"


def get_field_to_component(species_names: Iterable[str]) -> dict[str, dict[str, int]]:
    field_to_component: dict[str, dict[str, int]] = {}
    field_to_component["jeh"] = {
        "jx_ec": 0,
        "jy_ec": 1,
        "jz_ec": 2,
        "ex_ec": 3,
        "ey_ec": 4,
        "ez_ec": 5,
        "hx_fc": 6,
        "hy_fc": 7,
        "hz_fc": 8,
    }
    field_to_component["dive"] = {"dive": 0}
    field_to_component["rho"] = {"rho": 0}
    field_to_component["d_rho"] = {"d_rho": 0}
    field_to_component["div_j"] = {"div_j": 0}

    # keeping 'all_1st' for backwards compatibility
    field_to_component["all_1st"] = {}
    field_to_component["all_1st_cc"] = {}
    moments = [
        "rho",
        "jx",
        "jy",
        "jz",
        "px",
        "py",
        "pz",
        "txx",
        "tyy",
        "tzz",
        "txy",
        "tyz",
        "tzx",
    ]
    for species_idx, species_name in enumerate(species_names):
        for moment_idx, moment in enumerate(moments):
            field_to_component["all_1st"][f"{moment}_{species_name}"] = moment_idx + 13 * species_idx
            field_to_component["all_1st_cc"][f"{moment}_{species_name}"] = moment_idx + 13 * species_idx

    return field_to_component

from __future__ import annotations

from collections.abc import Generator, Hashable, Iterable
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray


class RunInfo:
    """Global information about the PSC run

    Currently stores domain info.
    TODO: Should also know about timestep, species, whatever...
    """

    def __init__(
        self,
        ds: xr.Dataset,
        length: ArrayLike | None = None,
        corner: ArrayLike | None = None,
    ) -> None:
        first_var = ds[next(iter(ds))]
        self.gdims = np.asarray(first_var.shape)[::-1][:3]

        self.length = ds.attrs.get("length", length)
        self.corner = ds.attrs.get("corner", corner)

        if self.length is None:
            message = "Dataset is missing length. A value must be manually provided."
            raise ValueError(message)
        if self.corner is None:
            message = "Dataset is missing corner. A value must be manually provided."
            raise ValueError(message)

        self.x = self._get_coord(0)
        self.y = self._get_coord(1)
        self.z = self._get_coord(2)

    def _get_coord(self, coord_idx: int) -> NDArray[Any]:
        return np.linspace(
            start=self.corner[coord_idx],
            stop=self.corner[coord_idx] + self.length[coord_idx],
            num=self.gdims[coord_idx],
            endpoint=False,
        )

    def __repr__(self) -> str:
        return f"Psc(gdims={self.gdims}, length={self.length}, corner={self.corner})"


def iter_components(field: Hashable, species_names: Iterable[str]) -> Generator[str]:
    if field == "jeh":
        yield from ["jx_ec", "jy_ec", "jz_ec", "ex_ec", "ey_ec", "ez_ec", "hx_fc", "hy_fc", "hz_fc"]  # fmt: off
    elif field in ["dive", "rho", "d_rho", "dt_divj"]:
        yield str(field)
    elif field in ["all_1st", "all_1st_cc"]:
        moments = ["rho", "jx", "jy", "jz", "px", "py", "pz", "txx", "tyy", "tzz", "txy", "tyz", "tzx"]  # fmt: off
        for species_name in species_names:
            for moment in moments:
                yield f"{moment}_{species_name}"


def unwrap_float(arr: np.ndarray) -> float:
    if arr.ndim == 0:
        return float(arr)
    return float(arr[0])


def decode_psc(
    ds: xr.Dataset,
    species_names: Iterable[str],
    length: ArrayLike | None = None,
    corner: ArrayLike | None = None,
) -> xr.Dataset:
    dims = list(ds.dims)
    if "dim_0_1" in dims:
        # for compatibility, if dimensions weren't saved as attribute in the .bp file,
        # fix them up here
        ds = ds.rename_dims(
            {
                dims[0]: "step",
                dims[1]: "component",
                dims[2]: "z",
                dims[3]: "y",
                dims[4]: "x",
            }
        )
    ds = ds.squeeze("step")

    for var_name in ds:
        components = list(iter_components(var_name, species_names))
        for component_idx, component in enumerate(components):
            ds = ds.assign({component: ds[var_name].isel(component=component_idx)})
        if var_name not in components:
            ds = ds.drop_vars([var_name])

    run_info = RunInfo(ds, length=length, corner=corner)
    coords = {
        "x": ("x", run_info.x),
        "y": ("y", run_info.y),
        "z": ("z", run_info.z),
        "t": unwrap_float(ds.attrs["time"]),
    }
    ds = ds.assign_coords(coords)

    return ds

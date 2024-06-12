from __future__ import annotations

import io
import os
from typing import Any, Iterable, override

import xarray
from numpy.typing import ArrayLike
from xarray.backends import CachingFileManager
from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    _normalize_path,
)
from xarray.backends.locks import (
    SerializableLock,
    combine_locks,
    ensure_lock,
    get_write_lock,
)
from xarray.core import indexing
from xarray.core.datatree import DataTree
from xarray.core.utils import FrozenDict

from . import adios2py
from .psc import field_to_component, RunInfo

# adios2 is not thread safe
ADIOS2_LOCK = SerializableLock()


class PscAdios2Array(BackendArray):
    """Lazy evaluation of a variable stored in PSC's adios2 field output.

    This also takes care of slicing out the specific component of the data stored as 4-d array.
    """

    def __init__(self, variable_name, datastore, orig_varname, component):
        self.variable_name = variable_name
        self.datastore = datastore
        self._orig_varname = orig_varname
        self._component = component
        array = self.get_array()
        self.shape = array.shape[:-1]
        self.dtype = array.dtype

    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        return ds[self._orig_varname]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.BASIC, self._getitem)

    def _getitem(self, args):
        with self.datastore.lock:
            array = self.get_array(needs_lock=False)
            return array[(*args, self._component)]  # FIXME add ... in between


class PscAdios2Store(AbstractDataStore):
    def __init__(
        self,
        manager,
        species_names,
        mode=None,
        lock=ADIOS2_LOCK,
        length=None,
        corner=None,
    ):
        self._manager = manager
        self._mode = mode
        self.lock = ensure_lock(lock)
        self.psc = RunInfo(self.ds, length=length, corner=corner)
        self._species_names = species_names

    @classmethod
    def open(cls, filename, species_names, mode="r", lock=None, length=None, corner=None):
        if lock is None:
            if mode == "r":
                lock = ADIOS2_LOCK
            else:
                lock = combine_locks([ADIOS2_LOCK, get_write_lock(filename)])

        manager = CachingFileManager(adios2py.File, filename, mode=mode)
        return cls(manager, species_names, mode=mode, lock=lock, length=length, corner=corner)

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = root
        return ds

    @property
    def ds(self):
        return self._acquire()

    @override
    def get_variables(self):
        fields_to_index = field_to_component(self._species_names)

        variables = {}
        for varname in self.ds.variables:
            for field, idx in fields_to_index[varname].items():
                variables[field] = (varname, idx)

        return FrozenDict((k, self.open_store_variable(k, v)) for k, v in variables.items())

    def open_store_variable(self, name, tpl):
        orig_varname, idx = tpl
        data = indexing.LazilyIndexedArray(PscAdios2Array(name, self, orig_varname, idx))
        dims = ["x", "y", "z"]
        coords = {"x": self.psc.x, "y": self.psc.y, "z": self.psc.z}
        return xarray.DataArray(data, dims=dims, coords=coords)

    @override
    def get_attrs(self):
        # FIXME this is not the best way to get attributes
        def expand_attr(attr):
            data = attr.Data()
            if len(data) == 1:
                return data[0]
            return data

        return FrozenDict((name, expand_attr(self.ds._io.InquireAttribute(name))) for name in self.ds.attributes)

    @override
    def get_dimensions(self):
        raise NotImplementedError()


def psc_open_dataset(
    filename_or_obj,
    species_names: list[str] | None = None,
    length: ArrayLike | None = None,
    corner: ArrayLike | None = None,
) -> xarray.Dataset:
    filename_or_obj = _normalize_path(filename_or_obj)
    store = PscAdios2Store.open(filename_or_obj, species_names, length=length, corner=corner)

    data_vars, attrs = store.load()
    ds = xarray.Dataset(data_vars=data_vars, attrs=attrs)
    ds.set_close(store.close)
    return ds


class PscAdios2BackendEntrypoint(BackendEntrypoint):
    available = True

    @override
    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | io.BufferedIOBase | AbstractDataStore,
        *,
        drop_variables: str | Iterable[str] | None = None,
        length: ArrayLike | None = None,
        corner: ArrayLike | None = None,
        species_names: Iterable[str] | None = None,  # e.g. ['e', 'i']; FIXME should be readable from file
        **kwargs: Any,
    ) -> xarray.Dataset:
        if not isinstance(filename_or_obj, (str, os.PathLike)):
            raise NotImplementedError()

        return psc_open_dataset(
            filename_or_obj,
            species_names,
            length=length,
            corner=corner,
        )

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    @override
    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".bp"}

    @override
    def open_datatree(self, filename_or_obj: str | os.PathLike[Any] | io.BufferedIOBase | AbstractDataStore, **kwargs: Any) -> DataTree:
        raise NotImplementedError()


if xarray.__version__ == "2023.4.1":
    # FIXME determine exactly when the API changed
    BACKEND_ENTRYPOINTS["pscadios2"] = ("psc", PscAdios2BackendEntrypoint)
else:
    # API of version 0.19.0
    BACKEND_ENTRYPOINTS["pscadios2"] = PscAdios2BackendEntrypoint

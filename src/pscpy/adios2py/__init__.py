from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any

import adios2
import adios2.stream
import numpy as np
from adios2.adios import Adios
from numpy.typing import ArrayLike, NDArray

_ad = Adios()


class Variable:
    """Wrapper for an `adios2.Variable` object to facilitate loading and indexing into it."""

    def __init__(self, var: adios2.Variable, engine: adios2.Engine) -> None:
        self._var = var
        self._engine = engine
        self.name = self._name()
        self.shape = self._shape()
        self.dtype = self._dtype()
        logging.debug("variable __init__ var %s engine %s", var, engine)

    def close(self) -> None:
        logging.debug("adios2py.variable close")
        self._var = None
        self._engine = None

    def _assert_not_closed(self) -> None:
        if not self._var:
            raise ValueError("adios2py: variable is closed")

    def _set_selection(self, start: ArrayLike, count: ArrayLike) -> None:
        self._assert_not_closed()

        self._var.set_selection((start[::-1], count[::-1]))

    def _shape(self) -> ArrayLike:
        self._assert_not_closed()

        return self._var.shape()[::-1]

    def _name(self) -> str:
        self._assert_not_closed()

        return self._var.name()  # type: ignore[no-any-return]

    def _dtype(self) -> np.dtype:
        self._assert_not_closed()

        return adios2.type_adios_to_numpy(self._var.type())

    def __getitem__(self, args: Any) -> NDArray:
        self._assert_not_closed()

        if not isinstance(args, tuple):
            args = (args,)

        shape = self.shape
        sel_start = np.zeros_like(shape)
        sel_count = np.zeros_like(shape)
        arr_shape = []

        for d, arg in enumerate(args):
            if isinstance(arg, slice):
                start, stop, step = arg.indices(shape[d])
                assert stop > start
                assert step == 1
                sel_start[d] = start
                sel_count[d] = stop - start
                arr_shape.append(sel_count[d])
                continue

            try:
                idx = int(arg)
            except ValueError:
                pass
            else:
                if idx < 0:
                    idx += shape[d]
                sel_start[d] = idx
                sel_count[d] = 1
                continue

            raise RuntimeError(f"invalid args to __getitem__: {args}")

        for d in range(len(args), len(shape)):
            sel_start[d] = 0
            sel_count[d] = shape[d]
            arr_shape.append(sel_count[d])

        self._set_selection(sel_start, sel_count)

        arr = np.empty(arr_shape, dtype=self.dtype, order="F")  # FIXME is column-major correct?
        self._engine.get(self._var, arr, adios2.bindings.Mode.Sync)
        return arr

    def __repr__(self) -> str:
        return f"adios2py.variable(name={self.name}, shape={self.shape}, dtype={self.dtype}"


class File:
    """Wrapper for an `adios2.IO` object to facilitate variable and attribute reading."""

    def __init__(self, filename: str, mode: str = "r") -> None:
        logging.debug("adios2py: __init__ %s", filename)
        assert mode == "r"
        self._io_name = f"io-{filename}"
        self._io = _ad.declare_io(self._io_name)
        self._engine = self._io.open(filename, adios2.bindings.Mode.Read)
        self._open_vars: dict[str, Variable] = {}

        self.variable_names: Collection[str] = self._io.available_variables().keys()
        self.attribute_names: Collection[str] = self._io.available_attributes().keys()

    def __enter__(self) -> File:
        logging.debug("adios2py: __enter__")
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        logging.debug("adios2py: __exit__")
        self.close()

    def __del__(self) -> None:
        logging.debug("adios2py: __del__")
        if self._engine:
            self.close()

    def close(self) -> None:
        logging.debug("adios2py: close")
        logging.debug("open vars %s", self._open_vars)
        for var in self._open_vars.values():
            var.close()

        self._engine.close()
        self._engine = None

        _ad.remove_io(self._io_name)
        self._io = None
        self._io_name = None

    def get_variable(self, variable_name: str) -> Variable:
        var = Variable(self._io.inquire_variable(variable_name), self._engine)
        self._open_vars[variable_name] = var
        return var

    def get_attribute(self, attribute_name: str) -> Any:
        adios2_attr = self._io.inquire_attribute(attribute_name)
        data = adios2_attr.data()
        # FIXME use SingleValue when writing data to avoid doing this (?)
        if len(data) == 1:
            return data[0]
        return data

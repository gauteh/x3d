from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.backends.common import AbstractDataStore
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict

from x3d import layout as costs
from x3d import url as x3d_url
from x3d._core import Array, Store

log = logging.getLogger("x3d")

EXACT_LAYOUTS = ("ts", "map")
LAYOUT_MODES = ("auto", "ts", "map", "ov")
# Coordinates that are 2-D georeferencing fields rather than data.
GEO_COORDS = ("latitude", "longitude")


@dataclass
class Run:
    """The metadata that defines one model run in the bucket."""

    root: str
    run: str
    info: dict  # dataset.json
    metas: dict[str, dict]  # layout -> zarr.json (consolidated metadata)
    _stores: dict[str, Store] = field(default_factory=dict)

    @property
    def location(self) -> str:
        return f"{self.root}/{self.run}"

    def layout_location(self, layout: str) -> str:
        return f"{self.location}/{layout}.zarr"


def _get_json(store: Store, key: str) -> dict:
    raw = store.get(key)
    if raw is None:
        raise FileNotFoundError(f"{store.location}/{key}")
    return json.loads(raw)


def open_run(root: str, run: str | None = None, layouts=EXACT_LAYOUTS) -> Run:
    """Resolve ``latest.json`` (unless ``run`` is given) and read the run metadata: ``dataset.json``
    and the consolidated metadata of each layout; a handful of small requests, made concurrently."""
    root_store = Store(root)
    if run is None:
        run = str(_get_json(root_store, "latest.json")["run"])
    rs = root_store.child(run)
    with ThreadPoolExecutor(len(layouts) + 1) as ex:
        info = ex.submit(_get_json, rs, "dataset.json")
        metas = {name: ex.submit(_get_json, rs, f"{name}.zarr/zarr.json") for name in layouts}
        return Run(root, run, info.result(), {n: f.result() for n, f in metas.items()})


def _arrays(meta: dict) -> dict[str, dict]:
    nodes = meta.get("consolidated_metadata", {}).get("metadata")
    if nodes is None:
        raise ValueError("store has no consolidated metadata")
    return {n: m for n, m in nodes.items() if m.get("node_type") == "array"}


def _clean_attrs(meta: dict) -> dict:
    attrs = dict(meta.get("attributes", {}))
    attrs.pop("_ChunkSizes", None)
    # zarr v2 style base64 NaN: the array fill_value already says it.
    if not isinstance(attrs.get("_FillValue", 0), (int, float)):
        del attrs["_FillValue"]
    # A float32 scale makes xarray decode int16 to float32 instead of float64: half the memory and bytes moved.
    if "scale_factor" in attrs:
        attrs["scale_factor"] = np.float32(attrs["scale_factor"])
    return attrs


def _plain_slice(s, n: int):
    """Normalise one basic index to ((start, stop, step>0), squeeze, flip)."""
    if isinstance(s, (int, np.integer)):
        i = int(s) + n if s < 0 else int(s)
        return (i, i + 1, 1), True, False
    a, b, st = s.indices(n)
    if st > 0:
        return (a, max(a, b), st), False, False
    count = len(range(a, b, st))
    if count == 0:
        return (0, 0, 1), False, False
    return (a + (count - 1) * st, a + 1, -st), False, True


class X3DArray(BackendArray):
    """A variable stored in one or more layouts of a run. Each read is planned per layout and
    served from the cheapest one (or from the single layout, when only one is requested)."""

    def __init__(self, spec: dict[str, tuple[str, str, str]], shape, dtype, unpack: dict | None = None):
        # spec: layout -> (store location, array path, zarr.json of the array)
        # unpack: {"scale_factor": ..., "fill_value": ...}: the core returns float32 (CF mask_and_scale done
        # per chunk, in parallel) instead of the stored int16
        self.spec = spec
        self.shape = tuple(shape)
        self.unpack = unpack
        self.dtype = np.dtype("float32" if unpack else dtype)
        self.last_layout: str | None = None
        self._arrays: dict[str, Array] | None = None

    @property
    def arrays(self) -> dict[str, Array]:
        if self._arrays is None:
            self._arrays = {n: Array(Store(loc), path, meta, **(self.unpack or {})) for n, (loc, path, meta) in self.spec.items()}
        return self._arrays

    def __getstate__(self):
        return {"spec": self.spec, "shape": self.shape, "dtype": self.dtype, "unpack": self.unpack}

    def __setstate__(self, state):
        self.__init__(state["spec"], state["shape"], state["dtype"], state["unpack"])

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(key, self.shape, indexing.IndexingSupport.BASIC, self._getitem)

    def pick(self, sel) -> str:
        arrays = self.arrays
        if len(arrays) == 1:
            return next(iter(arrays))
        return costs.choose({n: a.plan(sel) for n, a in arrays.items()})

    def _getitem(self, key):
        norm = [_plain_slice(k, n) for k, n in zip(key, self.shape)]
        sel = [s for s, _, _ in norm]
        layout = self.pick(sel)
        self.last_layout = layout
        log.debug("read %s via %s", sel, layout)
        raw = self.arrays[layout].read_raw(sel)
        data = raw.view(self.dtype).reshape([max(0, -(-(b - a) // st)) for a, b, st in sel])
        index = tuple(0 if sq else (slice(None, None, -1) if fl else slice(None)) for _, sq, fl in norm)
        return data[index]


class X3DStore(AbstractDataStore):
    def __init__(self, run: Run, mode: str, ov: dict | None = None, mask_and_scale: bool = True):
        self.run = run
        self.mode = mode
        self._ov = ov
        self.mask_and_scale = mask_and_scale

    def get_dimensions(self):
        return {d: n for v in self.get_variables().values() for d, n in zip(v.dims, v.shape)}

    def get_attrs(self):
        base = self.run.metas["ov" if self.mode == "ov" else "ts"].get("attributes", {})
        attrs = dict(base)
        attrs.update(x3d_run=self.run.run, x3d_model_run=self.run.info.get("model_run", ""), x3d_layout=self.mode)
        return FrozenDict(attrs)

    def get_variables(self):
        run = self.run
        times = np.array([t.rstrip("Z") for t in run.info["times"]], dtype="datetime64[ns]")
        out = {"time": xr.Variable(("time",), times, {"standard_name": "time"})}

        if self.mode == "ov":
            layouts = ("ov",)
            skip = {"time", *GEO_COORDS}
        else:
            layouts = EXACT_LAYOUTS if self.mode == "auto" else (self.mode,)
            skip = {"time"}

        per_layout = {n: _arrays(run.metas[n]) for n in layouts}
        for name, meta in per_layout[layouts[0]].items():
            if name in skip:
                continue
            spec = {
                n: (run.layout_location(n), name, json.dumps(per_layout[n][name]))
                for n in layouts
                if name in per_layout[n] and per_layout[n][name]["shape"] == meta["shape"]
            }
            dims = tuple(meta["dimension_names"])
            attrs, unpack, encoding = _clean_attrs(meta), None, {}
            if self.mask_and_scale and meta["data_type"] == "int16" and "scale_factor" in attrs:
                # unpacked by the core: the CF attributes move to encoding so that xarray does not decode again
                unpack = {"scale_factor": float(attrs.pop("scale_factor"))}
                encoding = {"scale_factor": np.float32(unpack["scale_factor"]), "dtype": np.dtype("int16")}
                if "_FillValue" in attrs:
                    unpack["fill_value"] = encoding["_FillValue"] = int(attrs.pop("_FillValue"))
            var = xr.Variable(dims, indexing.LazilyIndexedArray(X3DArray(spec, meta["shape"], meta["data_type"], unpack)), attrs, encoding)
            var.encoding["preferred_chunks"] = {d: min(c, s) for d, c, s in zip(dims, meta["chunk_grid"]["configuration"]["chunk_shape"], meta["shape"])}
            out[name] = var

        if self.mode == "ov":
            f = run.info["ov_factor"]
            for name in ("rlat", "rlon"):
                fine = np.asarray(_read_coord(run, name))
                out[name] = xr.Variable((name,), _block_mean(fine, f), _clean_attrs(_arrays(run.metas["ts"])[name]))
        return out

    def close(self):
        pass


def _read_coord(run: Run, name: str) -> np.ndarray:
    meta = _arrays(run.metas["ts"])[name]
    arr = Array(Store(run.layout_location("ts")), name, json.dumps(meta))
    return arr.read_raw([(0, meta["shape"][0], 1)]).view(meta["data_type"])


def _block_mean(x: np.ndarray, f: int) -> np.ndarray:
    """Mean over blocks of ``f`` (the last one may be partial), like the overview resampling."""
    starts = np.arange(0, len(x), f)
    return np.add.reduceat(x, starts) / np.diff(np.append(starts, len(x)))


class X3DBackendEntrypoint(BackendEntrypoint):
    description = "Open the waves.met.no Zarr store (x3d+https://host/bucket/ww3_4km)"
    url = "https://github.com/metno/x3d"
    open_dataset_parameters = [
        "filename_or_obj", "drop_variables", "mask_and_scale", "decode_times", "concat_characters",
        "decode_coords", "use_cftime", "decode_timedelta", "layout", "run",
    ]

    def guess_can_open(self, filename_or_obj):
        return x3d_url.is_x3d_url(os.fspath(filename_or_obj) if isinstance(filename_or_obj, os.PathLike) else filename_or_obj)

    def open_dataset(
        self, filename_or_obj, *, drop_variables=None, mask_and_scale=True, decode_times=True,
        concat_characters=True, decode_coords=True, use_cftime=None, decode_timedelta=None,
        layout=None, run=None,
    ):
        """``layout``: ``"auto"`` (default) reads each selection from the cheapest of the exact layouts
        (``ts`` for series, ``map`` for fields); ``"ts"``/``"map"`` pin one; ``"ov"`` opens the lossy
        1/4 resolution overview grid. ``run`` pins a run id instead of ``latest.json``."""
        root, opts = x3d_url.parse(os.fspath(filename_or_obj))
        mode = layout or opts.get("layout", "auto")
        if mode not in LAYOUT_MODES:
            raise ValueError(f"layout must be one of {LAYOUT_MODES}, not {mode!r}")
        # ts always: its metadata carries the global attributes and, for the overview, the fine coordinates
        needed = ("ts", *{"auto": ("map",), "map": ("map",), "ov": ("ov",), "ts": ()}[mode])
        r = open_run(root, run or opts.get("run"), needed)
        ds = StoreBackendEntrypoint().open_dataset(
            X3DStore(r, mode, mask_and_scale=mask_and_scale),
            mask_and_scale=mask_and_scale, decode_times=decode_times, concat_characters=concat_characters,
            decode_coords=decode_coords, drop_variables=drop_variables, use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )
        ds = ds.set_coords([c for c in GEO_COORDS if c in ds])
        ds.encoding["source"] = r.location
        return ds


def open_x3d(url: str, **kwargs) -> xr.Dataset:
    return xr.open_dataset(url, engine="x3d", **kwargs)

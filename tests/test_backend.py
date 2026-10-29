import logging
import pickle

import numpy as np
import pytest
import xarray as xr
from xarray.core import indexing

from x3d import layout, url
from x3d.backend import X3DArray

from conftest import NT, NX, NY, OVF, RUN, SCALES, packed, reference

SELECTIONS = {
    "point_series": dict(rlat=25, rlon=20),
    "field": dict(time=3),
    "box": dict(time=slice(1, 5), rlat=slice(10, 35), rlon=slice(12, 28)),
    "strided_reverse": dict(time=slice(5, 0, -2), rlat=slice(39, 3, -5), rlon=slice(2, 30, 7)),
    "land_only": dict(rlat=slice(0, 10), rlon=slice(0, 10)),
    "empty": dict(time=slice(2, 2)),
    "negative_int": dict(time=-1, rlat=-3),
}


def backing(da) -> X3DArray:
    """The X3DArray behind a (CF-decoded, lazily indexed) variable."""
    node = da.variable._data
    while not isinstance(node, X3DArray):
        node = node.array
    return node


def open_local(store_dir, **kw):
    return xr.open_dataset(f"x3d+file://{store_dir}", **kw)


def expected(name, sel):
    """What the packed fixture decodes to, from numpy alone."""
    v = packed(name).astype("float32")
    v[v == -32768] = np.nan
    da = xr.DataArray(v * np.float32(SCALES[name]), dims=("time", "rlat", "rlon"))
    return da.isel(sel).values


def test_url_parse():
    assert url.parse("x3d+https://data.met.no/ww3_4km") == ("https://data.met.no/ww3_4km", {})
    assert url.parse("x3d+https://h/b:c/ww3_4km/?run=1&layout=ts") == ("https://h/b:c/ww3_4km", {"run": "1", "layout": "ts"})
    assert url.parse("/data/ww3_4km") == ("/data/ww3_4km", {})


def test_guess_can_open_only_prefixed_urls():
    from x3d.backend import X3DBackendEntrypoint

    ep = X3DBackendEntrypoint()
    assert ep.guess_can_open("x3d+https://data.met.no/ww3_4km")
    assert not ep.guess_can_open("https://data.met.no/x.zarr")
    assert not ep.guess_can_open("file.nc")


def test_layout_choice_follows_access_pattern():
    # (requests, bytes): a series is one shard index + one chunk vs a request per time step
    assert layout.choose({"ts": (2, 36_000), "map": (144, 5_000_000)}) == "ts"
    assert layout.choose({"ts": (2925, 2e8), "map": (16, 2e6)}) == "map"


def test_open_metadata(store_dir):
    ds = open_local(store_dir)
    assert set(ds.data_vars) == set(SCALES)
    assert {"latitude", "longitude"} <= set(ds.coords)
    assert ds.sizes == {"time": NT, "rlat": NY, "rlon": NX}
    assert ds.time.values[1] == np.datetime64("2026-01-01T01:00:00")
    assert ds.attrs["x3d_run"] == RUN
    assert ds.hs.dtype == np.float32
    assert ds.hs.attrs["units"] == "m"
    assert "_ChunkSizes" not in ds.hs.attrs
    assert ds.hs.encoding["_FillValue"] == -32768


@pytest.mark.parametrize("name", SCALES)
@pytest.mark.parametrize("sel", SELECTIONS.values(), ids=SELECTIONS.keys())
@pytest.mark.parametrize("layout_", ["auto", "ts", "map"])
def test_values_match_numpy_and_zarr_python(store_dir, name, sel, layout_):
    ds = open_local(store_dir, layout=layout_)
    got = ds[name].isel(sel).values
    np.testing.assert_array_equal(got, expected(name, sel))
    # independent reference implementation of the sharded format
    ref = reference(store_dir, "ts" if layout_ != "map" else "map")[name].isel(sel)
    np.testing.assert_allclose(got, ref.values, rtol=1e-6)  # reference decodes to float64


def test_coordinates_are_exact(store_dir):
    ds = open_local(store_dir)
    ref = reference(store_dir)
    np.testing.assert_array_equal(ds.latitude.values, ref.latitude.values)
    np.testing.assert_array_equal(ds.rlon.values, ref.rlon.values)


def test_auto_picks_ts_for_series_and_map_for_fields(store_dir):
    ds = open_local(store_dir)
    arr = backing(ds.hs)  # LazilyIndexedArray -> X3DArray
    assert isinstance(arr, X3DArray)
    ds.hs.isel(rlat=25, rlon=20).values
    assert arr.last_layout == "ts"
    ds.hs.isel(time=3).values
    assert arr.last_layout == "map"


def test_pinned_layout_is_respected(store_dir):
    for pin in ("ts", "map"):
        ds = open_local(store_dir, layout=pin)
        ds.hs.isel(time=1, rlat=3, rlon=4).values
        assert backing(ds.hs).last_layout == pin


def test_grid_names_come_from_dataset_json(projected_store_dir):
    """NorKyst-like store: Y/X dims and lat/lon coordinates are declared in dataset.json, not assumed."""
    ds = open_local(projected_store_dir)
    assert ds.hs.dims == ("time", "Y", "X") and {"lat", "lon"} <= set(ds.coords)
    for sel in (dict(Y=25, X=20), dict(time=2), dict(time=slice(1, 5), Y=slice(10, 35), X=slice(12, 28))):
        np.testing.assert_array_equal(ds.hs.isel(sel).values, expected("hs", {k.replace("Y", "rlat").replace("X", "rlon"): v for k, v in sel.items()}))
    ov = open_local(projected_store_dir, layout="ov")
    oy, ox = -(-NY // OVF), -(-NX // OVF)
    assert ov.hs.shape == (NT, oy, ox) and ov.hs.dims == ("time", "Y", "X") and ov.Y.shape == (oy,)
    assert "lat" not in ov.variables


def test_overview_is_coarse_grid_with_block_mean_coordinates(store_dir):
    ds = open_local(store_dir, layout="ov")
    oy, ox = -(-NY // OVF), -(-NX // OVF)
    assert ds.hs.shape == (NT, oy, ox)
    assert "latitude" not in ds.variables
    fine = reference(store_dir).rlat.values
    np.testing.assert_allclose(ds.rlat.values[:-1], fine[: (oy - 1) * OVF].reshape(oy - 1, OVF).mean(1))
    np.testing.assert_allclose(ds.rlat.values[-1], fine[(oy - 1) * OVF :].mean())
    np.testing.assert_array_equal(ds.hs.isel(time=2).values, expected("hs", dict(time=2))[::OVF, ::OVF])


def test_engine_guessed_from_prefix(store_dir):
    """No engine= needed for x3d+ URLs; plain paths still need it."""
    a = xr.open_dataset(f"x3d+file://{store_dir}")
    b = xr.open_dataset(f"x3d+file://{store_dir}?layout=ts")
    assert a.encoding["source"] and b.attrs["x3d_layout"] == "ts"
    with pytest.raises(ValueError):
        xr.open_dataset(str(store_dir))


def test_bad_layout_rejected(store_dir):
    with pytest.raises(ValueError, match="layout"):
        open_local(store_dir, layout="nope")


def test_missing_run_and_store(tmp_path, store_dir):
    with pytest.raises(FileNotFoundError):
        xr.open_dataset(f"x3d+file://{tmp_path}")
    with pytest.raises(FileNotFoundError):
        open_local(store_dir, run="19990101T000000Z")


def test_pinned_run(store_dir):
    assert open_local(store_dir, run=RUN).attrs["x3d_run"] == RUN


def test_pickle_roundtrip_for_dask(store_dir):
    ds = open_local(store_dir)
    arr = backing(ds.hs)
    clone = pickle.loads(pickle.dumps(arr))
    sel = indexing.BasicIndexer((slice(0, NT), slice(0, NY), slice(0, NX)))
    np.testing.assert_array_equal(clone[sel], arr[sel])


def test_dask_chunks(store_dir):
    pytest.importorskip("dask")
    ds = open_local(store_dir, chunks={})
    np.testing.assert_array_equal(ds.hs.isel(time=1).values, expected("hs", dict(time=1)))


def test_debug_log_reports_layout(store_dir, caplog):
    ds = open_local(store_dir)
    with caplog.at_level(logging.DEBUG, logger="x3d"):
        ds.hs.isel(rlat=25, rlon=20).values
    assert "via ts" in caplog.text


def test_http_range_reads_and_request_economy(http_store, store_dir):
    base, log = http_store
    ds = xr.open_dataset(f"x3d+{base}")
    opened = list(log)
    # latest.json, dataset.json and the consolidated metadata of the two exact layouts, plus the small
    # index coordinates (indexes are loaded eagerly by xarray); no data chunks
    meta = {f"/{store_dir.name}/{p}" for p in ("latest.json", f"{RUN}/dataset.json", f"{RUN}/ts.zarr/zarr.json", f"{RUN}/map.zarr/zarr.json")}
    paths = [p for p, _ in opened]
    assert meta <= set(paths)
    assert all(p in meta or "/rlat/" in p or "/rlon/" in p for p in paths)
    log.clear()
    series = ds.hs.isel(rlat=25, rlon=20).values
    np.testing.assert_array_equal(series, expected("hs", SELECTIONS["point_series"]))
    assert log and all("/ts.zarr/" in p and r is not None for p, r in log)
    assert len(log) <= 2  # the shard index and one inner chunk
    log.clear()
    field = ds.hs.isel(time=3).values
    np.testing.assert_array_equal(field, expected("hs", dict(time=3)))
    assert all("/map.zarr/" in p and r is not None for p, r in log)
    assert all("/c/3/" in p for p, _ in log)  # only that time step's shards
    ds.hs.isel(rlat=25, rlon=20).values
    log.clear()
    ds.hs.isel(rlat=25, rlon=20).values
    assert len(log) == 1  # shard index now cached; only the chunk


def test_http_missing_shards_read_as_fill(http_store):
    base, _ = http_store
    ds = xr.open_dataset(f"x3d+{base}", layout="map")
    assert np.isnan(ds.hs.isel(rlat=slice(0, 16), rlon=slice(0, 16)).values).all()


@pytest.mark.rgw
def test_rgw_open_and_layouts_agree(rgw_url):
    ds = xr.open_dataset(f"x3d+{rgw_url}")
    assert {"hs", "phs0", "phs1"} <= set(ds.data_vars)
    assert ds.sizes["time"] > 1
    assert ds.time.diff("time").dt.total_seconds().min() > 0
    ts = xr.open_dataset(f"x3d+{rgw_url}?run={ds.attrs['x3d_run']}&layout=ts")
    mp = xr.open_dataset(f"x3d+{rgw_url}?run={ds.attrs['x3d_run']}&layout=map")
    box = dict(time=slice(0, 4), rlat=slice(400, 440), rlon=slice(200, 260))
    a = ts.hs.isel(box).values
    np.testing.assert_array_equal(a, mp.hs.isel(box).values)
    assert np.isfinite(a).any()
    series = ds.hs.isel(rlat=500, rlon=300)
    np.testing.assert_array_equal(series.values, ts.hs.isel(rlat=500, rlon=300).values)
    assert backing(ds.hs).last_layout == "ts"


@pytest.mark.rgw
def test_rgw_overview_is_cheap_and_close_to_full_resolution(rgw_url):
    ov = xr.open_dataset(f"x3d+{rgw_url}?layout=ov")
    full = xr.open_dataset(f"x3d+{rgw_url}")
    assert ov.sizes["rlat"] * OVF >= full.sizes["rlat"] > (ov.sizes["rlat"] - 1) * OVF
    a, b = ov.hs.isel(time=0).values, full.hs.isel(time=0, rlat=slice(0, None, OVF), rlon=slice(0, None, OVF)).values
    both = np.isfinite(a) & np.isfinite(b)
    assert both.any() and abs(np.mean(a[both]) - np.mean(b[both])) < 0.5


@pytest.mark.store
def test_local_store_matches_zarr_python(local_store):
    ds = xr.open_dataset(f"x3d+file://{local_store}")
    run = ds.attrs["x3d_run"]
    ref = xr.open_zarr(local_store / run / "ts.zarr", consolidated=True)
    nt, ny, nx = ref.hs.shape
    # picked relative to the grid, so a small synthetic store (x3d-publisher's CI) works as well as the real one
    for sel in (
        dict(rlat=ny // 2, rlon=nx // 2), dict(time=nt // 7),
        dict(time=slice(nt - 12, 3, -7), rlat=slice(ny * 7 // 8, ny // 10, -37), rlon=slice(5, nx * 19 // 20, 11)),
    ):
        np.testing.assert_allclose(ds.hs.isel(sel).values, ref.hs.isel(sel).values.astype("float32"), rtol=1e-6, equal_nan=True)

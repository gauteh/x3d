"""THREDDS (OPeNDAP), the NetCDF file on /lustre and x3d (Zarr on RGW, and a local store) for the same hs data, with a plot.

    pytest --benchmark-enable --benchmark-only    # measure; writes docs/thredds_vs_x3d.png
    pytest --benchmark-enable --benchmark-only --benchmark-json=raw.json

Without --benchmark-enable (the default, see addopts) each case runs once as a plain test, no timing or plot.

Each round starts from a freshly opened dataset (not timed), so the timed call is a cold read: for x3d
that includes fetching the shard indexes. THREDDS reads the same time window as the RGW store (the steps of its
latest run, located in the aggregate by time); the lustre file and the local store read their first steps (the same
amount of data). For the file-based backends only the first round is cold: later rounds hit the OS page cache.

Backends: thredds; lustre (netCDF4 on the newest ww3_4km_<date>T00Z.nc below X3D_LUSTRE_ROOT, or X3D_LUSTRE_FILE);
x3d (RGW); x3d-local (x3d+file:// on the wave-world store, X3D_TEST_STORE). Unavailable ones are skipped.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from conftest import RGW_URL, WAVE_WORLD_STORE

pytestmark = pytest.mark.rgw

THREDDS_URL = os.environ.get("X3D_THREDDS_URL", "https://thredds.met.no/thredds/dodsC/ww3_4km_agg")
LUSTRE_ROOT = Path(os.environ.get("X3D_LUSTRE_ROOT", "/lustre/arkivB/projects/metproduction/DNMI_WAVE"))
OUT = Path(os.environ.get("X3D_BENCH_OUT", Path(__file__).parents[1] / "docs"))
ROUNDS = int(os.environ.get("X3D_BENCH_ROUNDS", 5))

BOX = dict(rlat=slice(400, 500), rlon=slice(200, 300))
# A fresh seed per run: a fixed one would hit the same points on THREDDS every time, and it caches them.
SEED = int(os.environ.get("X3D_BENCH_SEED", np.random.SeedSequence().entropy % 2**32))
RNG = np.random.default_rng(SEED)


def random_series(ds):
    """A different point every round, so server-side caches of the fixed point do not help."""
    return ds.hs.isel(rlat=int(RNG.integers(ds.sizes["rlat"])), rlon=int(RNG.integers(ds.sizes["rlon"]))).values


SCENARIOS = {
    "open": None,
    "series: 1 point, all steps": lambda ds: ds.hs.isel(rlat=500, rlon=300).values,
    "series: random point, all steps": random_series,
    "field: 1 step, full grid": lambda ds: ds.hs.isel(time=10).values,
    "box: 24 steps, 100x100": lambda ds: ds.hs.isel(time=slice(0, 24), **BOX).values,
    "fields: 6 steps, full grid": lambda ds: ds.hs.isel(time=slice(0, 6)).values,
}
BACKENDS = ("thredds", "lustre", "x3d", "x3d-local")
LABELS = {"thredds": "THREDDS OPeNDAP", "lustre": "netCDF4 on /lustre\n(uncompressed float32, 8.8 GB)", "x3d": "x3d (Zarr on RGW)\n(int16 + zstd, ~0.2 GB)", "x3d-local": "x3d (local store)"}
COLORS = {"thredds": "#7570b3", "lustre": "#d95f02", "x3d": "#1b9e77", "x3d-local": "#66a61e"}

results: dict[tuple[str, str], float] = {}


@pytest.fixture(scope="module")
def window(rgw_url):
    """(number of common steps, index of the first in the THREDDS aggregate); the store starts at its first."""
    x = xr.open_dataset(f"x3d+{rgw_url}")
    t = open_thredds()
    seconds = (x.time.values - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
    i0 = int(np.searchsorted(t.time.values, seconds[0]))
    # the aggregate may lag the newest run: use the steps both have
    n = min(len(seconds), t.sizes["time"] - i0)
    if n < 24 or not np.array_equal(t.time.values[i0 : i0 + n], seconds[:n]):
        pytest.skip("store time window not found in the THREDDS aggregate")
    return n, i0


def open_thredds():
    """THREDDS drops connections now and then; retry, then skip rather than fail."""
    err = None
    for _ in range(3):
        try:
            return xr.open_dataset(THREDDS_URL, engine="netcdf4", decode_times=False)
        except OSError as e:
            err = e
            time.sleep(1)
    pytest.skip(f"THREDDS not reachable: {err}")


def lustre_file() -> Path:
    """The newest hourly ww3_4km file (they are laid out as <root>/<year>/<month>/<day>/)."""
    if os.environ.get("X3D_LUSTRE_FILE"):
        found = [Path(os.environ["X3D_LUSTRE_FILE"])]
    else:
        found = sorted(LUSTRE_ROOT.glob("*/*/*/ww3_4km_????????T??Z.nc"), key=lambda f: f.name)
    if not found or not found[-1].exists():
        pytest.skip(f"no ww3_4km file below {LUSTRE_ROOT} (set X3D_LUSTRE_ROOT or X3D_LUSTRE_FILE)")
    return found[-1]


def open_backend(name: str, window) -> xr.Dataset:
    n, i0 = window
    if name == "x3d":
        return xr.open_dataset(f"x3d+{RGW_URL}")[["hs"]].isel(time=slice(0, n))
    if name == "x3d-local":
        if not (WAVE_WORLD_STORE / "latest.json").exists():
            pytest.skip(f"no local store at {WAVE_WORLD_STORE} (set X3D_TEST_STORE)")
        ds = xr.open_dataset(f"x3d+file://{WAVE_WORLD_STORE}")[["hs"]]
        if ds.sizes["time"] < 24:
            pytest.skip("local store has fewer than 24 steps")
        return ds.isel(time=slice(0, n))
    if name == "lustre":
        ds = xr.open_dataset(lustre_file(), engine="netcdf4", decode_times=False)[["hs"]]
        return ds.isel(time=slice(0, n))
    ds = open_thredds()[["hs"]]
    return ds.isel(time=slice(i0, i0 + n))


@pytest.fixture(scope="module", autouse=True)
def plot_at_end():
    yield
    if not results:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s for s in SCENARIOS if any((b, s) in results for b in BACKENDS)]
    fig, ax = plt.subplots(figsize=(11, 5))
    shown = [b for b in BACKENDS if any((b, s) in results for s in names)]
    w = 0.8 / len(shown)
    for k, b in enumerate(shown):
        xs = np.arange(len(names)) + (k - (len(shown) - 1) / 2) * w
        vals = [results.get((b, s), np.nan) for s in names]
        bars = ax.bar(xs, vals, w, label=LABELS[b], color=COLORS[b],
                      hatch="//" if b.startswith("x3d") else None, edgecolor="white" if b.startswith("x3d") else None)
        for r, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(r.get_x() + r.get_width() / 2, v, f"{v * 1e3:.1f}" if v < 0.1 else f"{v * 1e3:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("median time (s, log scale)")
    ax.set_xticks(range(len(names)), [n.replace(": ", ":\n") for n in names], fontsize=8)
    ax.set_title("ww3_4km hs: median time per access pattern (labels in ms)")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "thredds_vs_x3d.png", dpi=150)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_bench(benchmark, window, backend, scenario):
    benchmark.group = scenario
    read = SCENARIOS[scenario]
    if read is None:
        benchmark.pedantic(lambda: open_backend(backend, window), rounds=ROUNDS)
    else:
        benchmark.pedantic(lambda ds: read(ds), setup=lambda: ((open_backend(backend, window),), {}), rounds=ROUNDS)
    if benchmark.stats is not None:  # None when benchmarking is disabled
        results[(backend, scenario)] = benchmark.stats.stats.median


def test_thredds_and_x3d_agree(window):
    """The comparison is only meaningful if both return the same numbers (x3d is int16-packed)."""
    a = open_backend("thredds", window).hs.isel(time=slice(0, 24), **BOX).values
    b = open_backend("x3d", window).hs.isel(time=slice(0, 24), **BOX).values
    assert np.array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, atol=0.006, equal_nan=True)


def test_lustre_and_x3d_agree(rgw_url):
    """Same numbers from the file and from the store, when the store was made from that model run."""
    f = lustre_file()
    nc = xr.open_dataset(f, engine="netcdf4")
    x = xr.open_dataset(f"x3d+{rgw_url}")
    ref = np.datetime64(x.attrs["x3d_model_run"].rstrip("Z"), "ns")
    if nc["forecast_reference_time"].values != ref:
        pytest.skip(f"store is from model run {ref}, {f.name} from {nc['forecast_reference_time'].values}")
    common = np.intersect1d(nc.time.values, x.time.values)[:24]
    assert len(common) == 24
    a = nc.hs.sel(time=common).isel(**BOX).values
    b = x.hs.sel(time=common).isel(**BOX).values
    assert np.array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, atol=0.006, equal_nan=True)

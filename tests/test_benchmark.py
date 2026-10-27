"""THREDDS (OPeNDAP) vs x3d (Zarr on RGW) for the same hs data, with a plot.

    pytest --benchmark-enable --benchmark-only    # measure; writes docs/thredds_vs_x3d.png
    pytest --benchmark-enable --benchmark-only --benchmark-json=raw.json

Without --benchmark-enable (the default, see addopts) each case runs once as a plain test, no timing or plot.

Each round starts from a freshly opened dataset (not timed), so the timed call is a cold read: for x3d
that includes fetching the shard indexes. Both sides read the same time window (the steps of the store's latest run,
located in the THREDDS aggregate by time).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from conftest import RGW_URL

pytestmark = pytest.mark.rgw

THREDDS_URL = os.environ.get("X3D_THREDDS_URL", "https://thredds.met.no/thredds/dodsC/ww3_4km_agg")
OUT = Path(os.environ.get("X3D_BENCH_OUT", Path(__file__).parents[1] / "docs"))
ROUNDS = int(os.environ.get("X3D_BENCH_ROUNDS", 5))

BOX = dict(rlat=slice(400, 500), rlon=slice(200, 300))
RNG = np.random.default_rng(int(os.environ.get("X3D_BENCH_SEED", 0)))


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
BACKENDS = ("thredds", "x3d")

results: dict[tuple[str, str], float] = {}


@pytest.fixture(scope="module")
def window(rgw_url):
    """(number of common steps, index of the first in the THREDDS aggregate); the store starts at its first."""
    x = xr.open_dataset(f"x3d+{rgw_url}", engine="x3d")
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


def open_backend(name: str, window) -> xr.Dataset:
    n, i0 = window
    if name == "x3d":
        return xr.open_dataset(f"x3d+{RGW_URL}", engine="x3d")[["hs"]].isel(time=slice(0, n))
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
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.38
    colors = {"thredds": "#7570b3", "x3d": "#1b9e77"}
    for k, b in enumerate(BACKENDS):
        xs = np.arange(len(names)) + (k - 0.5) * w
        vals = [results.get((b, s), np.nan) for s in names]
        bars = ax.bar(xs, vals, w, label={"thredds": "THREDDS OPeNDAP", "x3d": "x3d (Zarr on RGW)"}[b], color=colors[b])
        for r, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(r.get_x() + r.get_width() / 2, v, f"{v * 1e3:.0f} ms", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("median time (s), cold read")
    ax.set_xticks(range(len(names)), [n.replace(": ", ":\n") for n in names], fontsize=8)
    ax.set_title("ww3_4km hs: THREDDS vs x3d")
    ax.legend()
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

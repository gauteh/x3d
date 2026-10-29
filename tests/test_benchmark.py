"""THREDDS (OPeNDAP), the NetCDF file on /lustre and x3d (Zarr on RGW, and a local store) for the same data, with a plot.

Two datasets, each in its own panel of the plot: ww3_4km `hs` (all four backends) and NorKyst-800 surface current
`u_eastward` (THREDDS and x3d on RGW).

    pytest --benchmark-enable --benchmark-only    # measure; writes docs/thredds_vs_x3d.png
    pytest --benchmark-enable --benchmark-only --benchmark-json=raw.json

Without --benchmark-enable (the default, see addopts) each case runs once as a plain test, no timing or plot.

Each round starts from a freshly opened dataset (not timed), so the timed call is a cold read: for x3d
that includes fetching the shard indexes. THREDDS reads the same time window as the RGW store (the steps of its
latest run, located in the aggregate by time); the lustre file and the local store read their first steps (the same
amount of data). For the file-based backends only the first round is cold: later rounds hit the OS page cache.

Backends: thredds; lustre (ww3_4km only: netCDF4 on the newest ww3_4km_<date>T00Z.nc below X3D_LUSTRE_ROOT, or
X3D_LUSTRE_FILE); x3d (RGW); x3d-local (ww3_4km only: x3d+file:// on a local store, X3D_TEST_STORE).
Unavailable ones are skipped.
"""

from __future__ import annotations

import functools
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from conftest import RGW_URL, LOCAL_STORE

pytestmark = pytest.mark.rgw

THREDDS_URL = os.environ.get("X3D_THREDDS_URL", "https://thredds.met.no/thredds/dodsC/ww3_4km_agg")
NK800_URL = os.environ.get(
    "X3D_RGW_NK800_URL", "https://rgw.met.no/5e4f32366fc74342866b4c67dbd4713d:norkyst800m-v3/norkyst800"
)
NK800_THREDDS_URL = os.environ.get(
    "X3D_THREDDS_NK800_URL", "https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_800m_m00_be"
)
LUSTRE_ROOT = Path(os.environ.get("X3D_LUSTRE_ROOT", "/lustre/arkivB/projects/metproduction/DNMI_WAVE"))
OUT = Path(os.environ.get("X3D_BENCH_OUT", Path(__file__).parents[1] / "docs"))
ROUNDS = int(os.environ.get("X3D_BENCH_ROUNDS", 5))

# `point` and `box` are (y, x) index (ranges) in the ocean; `atol` is the int16 packing precision of the variable
DATASETS = {
    "ww3_4km": dict(
        var="hs", y="rlat", x="rlon", rgw=RGW_URL, thredds=THREDDS_URL, point=(500, 300),
        box=(slice(400, 500), slice(200, 300)), atol=0.006, title="ww3_4km: significant wave height (hs)",
        backends=("thredds", "lustre", "x3d", "x3d-local"),
        labels={"lustre": "netCDF4 on /lustre\n(uncompressed float32, 8.8 GB)", "x3d": "x3d (Zarr on RGW)\n(int16 + zstd, ~0.2 GB)"},
    ),
    "norkyst800": dict(
        var="u_eastward", y="Y", x="X", rgw=NK800_URL, thredds=NK800_THREDDS_URL, point=(600, 300),
        box=(slice(400, 500), slice(200, 300)), atol=0.0006, title="NorKyst-800: surface current (u_eastward)",
        backends=("thredds", "x3d"), labels={"x3d": "x3d (Zarr on RGW)\n(int16 + zstd)"},
    ),
}
# A fresh seed per run: a fixed one would hit the same points on THREDDS every time, and it caches them.
SEED = int(os.environ.get("X3D_BENCH_SEED", np.random.SeedSequence().entropy % 2**32))
RNG = np.random.default_rng(SEED)

SCENARIOS = (
    "open",
    "series: 1 point, all steps",
    "series: random point, all steps",
    "field: 1 step, full grid",
    "box: 24 steps, 100x100",
    "fields: 6 steps, full grid",
)
LABELS = {"thredds": "THREDDS OPeNDAP", "lustre": "netCDF4 on /lustre", "x3d": "x3d (Zarr on RGW)", "x3d-local": "x3d (local store)"}
COLORS = {"thredds": "#7570b3", "lustre": "#d95f02", "x3d": "#1b9e77", "x3d-local": "#66a61e"}

results: dict[tuple[str, str, str], float] = {}


def read_scenario(dataset: str, scenario: str):
    """The read for `scenario` on `dataset`, as a function of the opened dataset (None: only open it)."""
    s = DATASETS[dataset]
    var, y, x = s["var"], s["y"], s["x"]
    box = {y: s["box"][0], x: s["box"][1]}

    def random_series(ds):
        """A different point every round, so server-side caches of the fixed point do not help."""
        return ds[var].isel({y: int(RNG.integers(ds.sizes[y])), x: int(RNG.integers(ds.sizes[x]))}).values

    return {
        "open": None,
        "series: 1 point, all steps": lambda ds: ds[var].isel({y: s["point"][0], x: s["point"][1]}).values,
        "series: random point, all steps": random_series,
        "field: 1 step, full grid": lambda ds: ds[var].isel(time=10).values,
        "box: 24 steps, 100x100": lambda ds: ds[var].isel(time=slice(0, 24), **box).values,
        "fields: 6 steps, full grid": lambda ds: ds[var].isel(time=slice(0, 6)).values,
    }[scenario]


@functools.cache
def get_window(dataset: str):
    """(number of common steps, index of the first in the THREDDS aggregate); the store starts at its first."""
    rgw = DATASETS[dataset]["rgw"]
    try:
        urllib.request.urlopen(f"{rgw}/latest.json", timeout=10).read()
    except Exception as e:  # network or bucket unavailable
        pytest.skip(f"RGW not reachable: {e}")
    x = xr.open_dataset(f"x3d+{rgw}")
    t = open_thredds(dataset)
    seconds = (x.time.values - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
    i0 = int(np.searchsorted(t.time.values, seconds[0]))
    # the aggregate may lag the newest run: use the steps both have
    n = min(len(seconds), t.sizes["time"] - i0)
    if n < 24 or not np.array_equal(t.time.values[i0 : i0 + n], seconds[:n]):
        pytest.skip("store time window not found in the THREDDS aggregate")
    return n, i0


@pytest.fixture
def window(dataset):
    return get_window(dataset)


def open_thredds(dataset: str):
    """THREDDS drops connections now and then; retry, then skip rather than fail."""
    err = None
    for _ in range(3):
        try:
            return xr.open_dataset(DATASETS[dataset]["thredds"], engine="netcdf4", decode_times=False)
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


def open_backend(dataset: str, name: str, window) -> xr.Dataset:
    """The variable of `dataset` (surface layer) over the common time window, opened afresh."""
    s = DATASETS[dataset]
    n, i0 = window
    if name == "x3d":
        return xr.open_dataset(f"x3d+{s['rgw']}")[[s["var"]]].isel(time=slice(0, n))
    if name == "x3d-local":
        if not (LOCAL_STORE / "latest.json").exists():
            pytest.skip(f"no local store at {LOCAL_STORE} (set X3D_TEST_STORE)")
        ds = xr.open_dataset(f"x3d+file://{LOCAL_STORE}")[[s["var"]]]
        if ds.sizes["time"] < 24:
            pytest.skip("local store has fewer than 24 steps")
        return ds.isel(time=slice(0, n))
    if name == "lustre":
        ds = xr.open_dataset(lustre_file(), engine="netcdf4", decode_times=False)[[s["var"]]]
        return ds.isel(time=slice(0, n))
    ds = open_thredds(dataset)[[s["var"]]]
    if "depth" in ds.dims:  # the x3d store is the surface layer (depth 0)
        ds = ds.isel(depth=0)
    return ds.isel(time=slice(i0, i0 + n))


@pytest.fixture(scope="module", autouse=True)
def plot_at_end():
    yield
    if not results:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown_sets = [d for d in DATASETS if any(k[0] == d for k in results)]
    fig, axes = plt.subplots(1, len(shown_sets), figsize=(8.5 * len(shown_sets), 5.8), sharey=True, squeeze=False)
    for p, (ax, d) in enumerate(zip(axes[0], shown_sets)):
        s = DATASETS[d]
        names = [sc for sc in SCENARIOS if any((d, b, sc) in results for b in s["backends"])]
        shown = [b for b in s["backends"] if any((d, b, sc) in results for sc in names)]
        w = 0.8 / len(shown)
        for k, b in enumerate(shown):
            xs = np.arange(len(names)) + (k - (len(shown) - 1) / 2) * w
            vals = [results.get((d, b, sc), np.nan) for sc in names]
            x3d = b.startswith("x3d")
            bars = ax.bar(xs, vals, w, label=s["labels"].get(b, LABELS[b]), color=COLORS[b],
                          hatch="//" if x3d else None, edgecolor="white" if x3d else None)
            for r, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(r.get_x() + r.get_width() / 2, v, f"{v * 1e3:.1f}" if v < 0.1 else f"{v * 1e3:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_yscale("log")
        if p == 0:
            ax.set_ylabel("median time (s, log scale)")
        ax.set_xticks(range(len(names)), [n.replace(": ", ":\n") for n in names], fontsize=8)
        ax.set_title(f"({'abcdef'[p]}) {s['title']}\nmedian time per access pattern (labels in ms)", fontsize=10)
        ax.grid(axis="y", which="both", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "thredds_vs_x3d.png", dpi=150)


def cases():
    return [(d, b) for d in DATASETS for b in DATASETS[d]["backends"]]


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("dataset,backend", cases())
def test_bench(benchmark, window, dataset, backend, scenario):
    benchmark.group = f"{dataset}: {scenario}"
    read = read_scenario(dataset, scenario)
    if read is None:
        benchmark.pedantic(lambda: open_backend(dataset, backend, window), rounds=ROUNDS)
    else:
        benchmark.pedantic(lambda ds: read(ds), setup=lambda: ((open_backend(dataset, backend, window),), {}), rounds=ROUNDS)
    if benchmark.stats is not None:  # None when benchmarking is disabled
        results[(dataset, backend, scenario)] = benchmark.stats.stats.median


@pytest.mark.parametrize("dataset", DATASETS)
def test_thredds_and_x3d_agree(dataset, window):
    """The comparison is only meaningful if both return the same numbers (x3d is int16-packed)."""
    s = DATASETS[dataset]
    if dataset == "norkyst800":  # the aggregate's past steps come from the newest run: only comparable to its store
        run = np.datetime64(xr.open_dataset(f"x3d+{s['rgw']}").attrs["x3d_model_run"].rstrip("Z"), "s")
        ref = open_thredds(dataset).forecast_reference_time.values
        if np.datetime64(int(ref), "s") != run:
            pytest.skip(f"store is from model run {run}, THREDDS from {np.datetime64(int(ref), 's')}")
    box = {s["y"]: s["box"][0], s["x"]: s["box"][1]}
    a = open_backend(dataset, "thredds", window)[s["var"]].isel(time=slice(0, 24), **box).values
    b = open_backend(dataset, "x3d", window)[s["var"]].isel(time=slice(0, 24), **box).values
    assert np.array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, atol=s["atol"], equal_nan=True)


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
    box = {"rlat": DATASETS["ww3_4km"]["box"][0], "rlon": DATASETS["ww3_4km"]["box"][1]}
    a = nc.hs.sel(time=common).isel(**box).values
    b = x.hs.sel(time=common).isel(**box).values
    assert np.array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, atol=0.006, equal_nan=True)

"""The Sentinel-3A along-track store on RGW has the same samples as the source files on /lustre.

The store is one 1-D series cut from the files of 3 h windows (the newest file of each window; the publisher's
`track.py`). A few windows spread over the whole history are compared: times, positions (longitude wrapped to
-180..180) and the variables, to the packing precision. Skipped where /lustre or RGW is not available."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytestmark = pytest.mark.rgw

S3A_URL = os.environ.get("X3D_RGW_S3A_URL", "https://rgw.met.no/079d3efebae2494db2a13c49766989ab:altimeter/altimeter_s3a")
S3A_ROOT = Path(os.environ.get("X3D_LUSTRE_S3A_ROOT", "/lustre/storeA/project/fou/om/altimeter/s3a"))
VARS = ("VAVH", "VAVH_UNFILTERED", "WIND_SPEED")


@pytest.fixture(scope="module")
def info():
    try:
        return json.loads(urllib.request.urlopen(f"{S3A_URL}/track/dataset.json", timeout=20).read())
    except Exception as e:
        pytest.skip(f"RGW not reachable: {e}")


@pytest.fixture(scope="module")
def store(info):
    return xr.open_dataset(f"x3d+{S3A_URL}")


def files_of(w: dict) -> list[Path]:
    """The source files of the window `w` (<root>/<year>/<month>/..._<start>_<end>_<creation>.nc), oldest version first."""
    s = w["start"]
    return sorted(S3A_ROOT.glob(f"{s[:4]}/{s[4:6]}/global_vavh_l3_rt_s3a_{s}_*_*.nc"), key=lambda f: f.stem.rsplit("_", 1)[1])


def sample_windows(info: dict) -> list[int]:
    """The first window, some in between and one a day or so before the end (the newest are still being re-issued)."""
    n = len(info["windows"])
    return sorted({0, n // 4, n // 2, 3 * n // 4, n - 10})


def test_store_matches_the_source_files(info, store):
    if not S3A_ROOT.is_dir():
        pytest.skip(f"no source files at {S3A_ROOT} (set X3D_LUSTRE_S3A_ROOT)")
    ws = info["windows"]
    assert store.sizes["time"] == info["n"]
    checked = 0
    for i in sample_windows(info):
        w = ws[i]
        files = files_of(w)
        if not files:
            continue  # the source has been cleaned up
        if i < len(ws) - 4:  # older windows are final: the published file is the newest one
            assert files[-1].stem.rsplit("_", 1)[1] == w["version"], f"window {w['start']}: a newer file exists than the published"
        src = [f for f in files if f.stem.endswith(w["version"])]
        assert src, f"window {w['start']}: published version {w['version']} is not among {[f.name for f in files]}"
        with xr.open_dataset(src[0]) as ds:
            ds = ds[[*VARS, "latitude", "longitude"]].load()

        # the publisher keeps the samples in increasing time, a time once, and none from before the previous window's end
        t = ds.time.values.astype("datetime64[s]")
        order = np.argsort(t, kind="stable")
        last = store.time.values[w["offset"] - 1].astype("datetime64[s]") if w["offset"] else np.datetime64("0000", "s")
        keep = order[np.append(True, np.diff(t[order].astype("int64")) > 0) & (t[order] > last)]
        assert len(keep) == w["n"], f"window {w['start']}: {len(keep)} usable samples in the file, {w['n']} published"

        got = store.isel(time=slice(w["offset"], w["offset"] + w["n"]))
        np.testing.assert_array_equal(got.time.values.astype("datetime64[s]"), t[keep])
        lon = (ds.longitude.values[keep] + 180) % 360 - 180
        dlon = (got.longitude.values - lon + 180) % 360 - 180  # the same longitude, whatever side of 180 it is on
        np.testing.assert_allclose(dlon, 0, atol=1e-6)
        np.testing.assert_allclose(got.latitude.values, ds.latitude.values[keep], atol=1e-6)
        for v, tol in zip(VARS, (0.0005, 0.0005, 0.005)):  # half the packing step
            a, b = ds[v].values[keep], got[v].values
            assert np.array_equal(np.isnan(a), np.isnan(b)), f"window {w['start']}: {v} has other gaps in the store"
            np.testing.assert_allclose(b, a, atol=tol, equal_nan=True, err_msg=f"window {w['start']}: {v}")
        checked += 1
    if not checked:
        pytest.skip(f"none of the sampled windows has source files below {S3A_ROOT}")

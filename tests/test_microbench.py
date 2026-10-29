"""Where does a read spend its time: fetching, decoding or copying?

    pytest tests/test_microbench.py -s -m "rgw or store"

`Array.read_profiled` times, per read and summed over the worker threads, the shard index requests, the
store fetches (HTTP round trip + body, or the file read), zstd decoding, and the copy into the output.
Compare the sums with `wall`: ``busy`` = (index + fetch + decode + scatter) / wall is how many threads
were kept busy on average. The local store takes the network out, which leaves decode + copy as the CPU cost.
Each read starts from a fresh `Array`, so shard indexes are cold. `X3D_THREADS=n` changes the pool size.
"""

from __future__ import annotations

import json
import os
import statistics

import pytest

from conftest import RGW_URL, LOCAL_STORE
from x3d._core import Array, Store
from x3d.backend import _arrays, open_run

ROUNDS = int(os.environ.get("X3D_BENCH_ROUNDS", 5))
KEYS = ("wall", "index", "fetch", "decode", "scatter")

# (layout, selection per dim of (time, rlat, rlon) as slices)
SCENARIOS = {
    "series: 1 point, all steps": ("ts", [(0, None, 1), (500, 501, 1), (300, 301, 1)]),
    "field: 1 step, full grid": ("map", [(10, 11, 1), (0, None, 1), (0, None, 1)]),
    "box: 24 steps, 100x100": ("ts", [(0, 24, 1), (400, 500, 1), (200, 300, 1)]),
    "fields: 6 steps, full grid": ("map", [(0, 6, 1), (0, None, 1), (0, None, 1)]),
}
rows: list[tuple] = []


def profile(root: str, layout: str, sel, var="hs") -> dict:
    run = open_run(root, layouts=(layout,))
    meta = _arrays(run.metas[layout])[var]
    sel = [(a, n if b is None else b, st) for (a, b, st), n in zip(sel, meta["shape"])]
    runs = []
    for _ in range(ROUNDS):
        arr = Array(Store(run.layout_location(layout)), var, json.dumps(meta))
        _, p = arr.read_profiled(sel)
        runs.append(p)
    med = {k: statistics.median(p[k] for p in runs) for k in KEYS}
    med["chunks"] = runs[0]["chunks"]
    med["MB"] = runs[0]["bytes_fetched"] / 1e6
    return med


@pytest.fixture(scope="module", autouse=True)
def report():
    yield
    if not rows:
        return
    head = f"{'':34}{'store':6}{'wall':>8}{'index':>8}{'fetch':>8}{'decode':>8}{'scatter':>8}{'busy':>6}{'chunks':>8}{'MB':>7}   (ms, medians of {ROUNDS})"
    print("\n" + head)
    for name, store, m in rows:
        busy = sum(m[k] for k in KEYS[1:]) / m["wall"]
        print(
            f"{name:34}{store:6}" + "".join(f"{m[k] * 1e3:8.1f}" for k in KEYS)
            + f"{busy:6.1f}{m['chunks']:8d}{m['MB']:7.2f}"
        )


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.rgw
def test_profile_rgw(rgw_url, scenario):
    layout, sel = SCENARIOS[scenario]
    m = profile(rgw_url, layout, sel)
    rows.append((scenario, "rgw", m))
    assert m["chunks"] > 0 and m["wall"] > 0


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.store
def test_profile_local(local_store, scenario):
    layout, sel = SCENARIOS[scenario]
    m = profile(str(LOCAL_STORE), layout, sel)
    rows.append((scenario, "local", m))
    assert m["chunks"] > 0 and m["wall"] > 0

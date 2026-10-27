"""Fixtures: a synthetic store with the wave-world layout (Zarr v3, sharded, ts/map/ov), a local
HTTP server with Range support that counts requests, and the optional real stores."""

from __future__ import annotations

import http.server
import json
import os
import threading
import zlib
import urllib.request
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.codecs import BytesCodec, ShardingCodec, ShardingCodecIndexLocation, ZstdCodec

NT, NY, NX, OVF = 6, 40, 30, 4
RUN = "20260101T000000Z"
SCALES = {"hs": 0.01, "tp": 0.1}
FILL = -32768
RGW_URL = os.environ.get(
    "X3D_RGW_URL", "https://rgw.met.no/5e4f32366fc74342866b4c67dbd4713d:waves-ww3/ww3_4km"
)
WAVE_WORLD_STORE = Path(os.environ.get("X3D_TEST_STORE", Path(__file__).parents[2] / "wave-world/store/ww3_4km"))


def packed(name: str) -> np.ndarray:
    """Deterministic int16 field with a land block (all fill) and some random fill."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    v = rng.integers(0, 1500, (NT, NY, NX)).astype("int16")
    v[:, :16, :16] = FILL  # whole chunks of land: absent from the shards
    v[:, 30, 5] = FILL
    return v


def _shard(shape, outer, inner, location):
    return dict(
        chunks=outer,
        serializer=ShardingCodec(
            chunk_shape=inner, codecs=[BytesCodec(), ZstdCodec()], index_location=location
        ),
        compressors=None,
    )


def _write_layout(path: Path, outer, inner, location):
    g = zarr.open_group(path, mode="w", zarr_format=3)
    g.attrs.update({"title": "synthetic", "Conventions": "CF-1.8"})
    g.create_array("time", data=np.arange(NT, dtype="int64"), chunks=(NT,), dimension_names=("time",),
                   attributes={"units": "hours since 2026-01-01 00:00:00"})
    rlat = np.linspace(-10, 10, NY)
    rlon = np.linspace(5, 25, NX)
    g.create_array("rlat", data=rlat, chunks=(NY,), dimension_names=("rlat",), attributes={"units": "degrees"})
    g.create_array("rlon", data=rlon, chunks=(NX,), dimension_names=("rlon",), attributes={"units": "degrees"})
    lat = (60 + rlat[:, None] + 0 * rlon[None, :]).astype("float32")
    lon = (rlon[None, :] + 0 * rlat[:, None]).astype("float32")
    for n, v, u in (("latitude", lat, "degrees_north"), ("longitude", lon, "degrees_east")):
        g.create_array(n, data=v, dimension_names=("rlat", "rlon"), attributes={"units": u},
                       **_shard(v.shape, outer[1:], inner[1:], location))
    for name, scale in SCALES.items():
        a = g.create_array(name, shape=(NT, NY, NX), dtype="int16", fill_value=FILL,
                           dimension_names=("time", "rlat", "rlon"),
                           attributes={"units": "m", "scale_factor": scale, "_FillValue": FILL, "_ChunkSizes": [1, NY, NX]},
                           **_shard((NT, NY, NX), outer, inner, location))
        a[:] = packed(name)
    zarr.consolidate_metadata(path)


def write_store(root: Path) -> Path:
    """<root>/latest.json and <root>/<run>/{dataset.json,ts.zarr,map.zarr,ov.zarr}."""
    run = root / RUN
    run.mkdir(parents=True)
    # ts: all time per shard, small spatial inner chunks; map: one time per shard, bigger spatial chunks
    _write_layout(run / "ts.zarr", (NT, 16, 16), (NT, 4, 4), ShardingCodecIndexLocation.start)
    _write_layout(run / "map.zarr", (1, 32, 32), (1, 16, 16), ShardingCodecIndexLocation.end)

    g = zarr.open_group(run / "ov.zarr", mode="w", zarr_format=3)
    g.create_array("time", data=np.arange(NT, dtype="int64"), chunks=(NT,), dimension_names=("time",))
    oy, ox = -(-NY // OVF), -(-NX // OVF)
    for name, scale in SCALES.items():
        a = g.create_array(name, shape=(NT, oy, ox), chunks=(NT, oy, ox), dtype="int16", fill_value=FILL,
                           dimension_names=("time", "rlat", "rlon"), compressors=ZstdCodec(),
                           attributes={"units": "m", "scale_factor": scale, "_FillValue": FILL})
        a[:] = packed(name)[:, ::OVF, ::OVF]
    zarr.consolidate_metadata(run / "ov.zarr")

    times = [f"2026-01-01T{h:02d}:00:00Z" for h in range(NT)]
    (run / "dataset.json").write_text(json.dumps(
        {"name": "synthetic", "run": RUN, "model_run": "2026-01-01T00:00:00Z", "times": times,
         "ny": NY, "nx": NX, "ov_factor": OVF, "scales": SCALES}))
    (root / "latest.json").write_text(json.dumps({"run": RUN}))
    return root


@pytest.fixture(scope="session")
def store_dir(tmp_path_factory) -> Path:
    return write_store(tmp_path_factory.mktemp("ww3_4km"))


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with single-range support; logs every request in ``server.log``."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        self.server.log.append((self.path, self.headers.get("Range")))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        rng = self.headers.get("Range")
        status, lo, hi = 200, 0, len(data) - 1
        if rng:
            spec = rng.removeprefix("bytes=")
            a, b = spec.split("-")
            lo, hi = (max(0, len(data) - int(b)), len(data) - 1) if not a else (int(a), min(int(b), len(data) - 1) if b else len(data) - 1)
            status = 206
        body = data[lo : hi + 1]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="session")
def http_store(store_dir):
    """(url of the store root, request log)."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(store_dir.parent)))
    srv.log = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/{store_dir.name}", srv.log
    srv.shutdown()


def reference(store_dir: Path, layout: str = "ts"):
    import xarray as xr

    return xr.open_zarr(store_dir / RUN / f"{layout}.zarr", consolidated=True, decode_times=False)


@pytest.fixture(scope="session")
def rgw_url():
    try:
        urllib.request.urlopen(f"{RGW_URL}/latest.json", timeout=10).read()
    except Exception as e:  # network or bucket unavailable
        pytest.skip(f"RGW not reachable: {e}")
    return RGW_URL


@pytest.fixture(scope="session")
def wave_world_store():
    if not (WAVE_WORLD_STORE / "latest.json").exists():
        pytest.skip(f"no wave-world store at {WAVE_WORLD_STORE} (set X3D_TEST_STORE)")
    return WAVE_WORLD_STORE

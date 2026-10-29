## usage
- [ ] does local store (x3d+file) work symmetrically to the s3 version? how does opening zarr distinguish between local and remote, is it possible to get rid of the +file or +https?.
- [x] engine in the URL: `xr.open_dataset("x3d+https://...")` works through `guess_can_open`; tested and used in the README and tests.
- [x] concurrent reads + parallel decompression: reads are already parallel (rayon pool, GIL released). `tests/test_microbench.py` (`Array.read_profiled`) shows that on RGW the time is the network (fetch summed over threads is 100s of ms; decode + copy are 2-8 ms even for 6 full fields), so no further decode/scatter work is worth it. Findings: a fresh `Store` per variable cost ~15 ms (new TLS connection + DNS), now one process-wide HTTP agent is shared (open 55 -> 15 ms, point series 17 -> 6 ms, field 23 -> 11 ms); added `X3D_THREADS`.
- [ ] remaining latency is round trips: cold shard-index fetch before every first chunk fetch (see the consolidated index sidecar below); box/field reads wait on ~10-50 parallel requests, so check coalescing neighbouring chunk ranges (a shard with `time=1` is one contiguous object).
- [ ] `ov` accessor on the variable: `da.x3d.ov` (and `ds.x3d.ov`) returns the overview for the same time/space selection, so `ds.hs.x3d.ov.isel(time=0).plot()` works. Needs the source URL/run on `var.encoding`.

## metadata + coordinate optimization
- [ ] fewer metadata fetches:
  - [ ] local disk cache for immutable objects: everything under `<run>/` (json, shard indexes, small coordinate arrays), never `latest.json`; `X3D_CACHE_DIR`, `X3D_CACHE=0`; disabled in the tests. A warm open would need only `latest.json`.
  - [ ] optional per-run consolidated index sidecar written by wave-world, with all shard indexes in one request. Estimated size: ~0.5 MB for ww3_4km, ~1.5 MB for NorKyst surface. Only worth it while it stays small.
  - [ ] derive regular `rlat`/`rlon` from `dataset.json` (start, step) instead of reading them.

## pipeline

- [x] (done: `../x3d-publisher`) eventually we need to decouple the pipeline from wave-world. i think it should live in its own repository. both x3d and wave-world will be users of the produced data (either local or remote/s3).

## norkyst
- [x] NorKyst-800 v3 (prefix `norkyst800` in the bucket, surface layer only), model `../x3d-publisher/src/x3d_publisher/models/norkyst800.toml`:
  - source: THREDDS best aggregate `fou-hi/norkystv3_800m_m00_be` (Y 1148 x X 2747, hourly, `depth=0.0`); window from the newest model run minus `back_hours = 48` to the end of the forecast (~169 steps). The old `norkyst800m-1h` ZDEPTHS files are stale.
  - variables: `u_eastward`, `v_northward`, `temperature`, `Uwind_eastward`, `Vwind_northward`; coordinates `lat`/`lon` (2-D).
  - [x] publisher: model-driven converter (`select`, `back_hours`, dims, lat/lon names), streams int16 batches to a spool, so float32 for the whole run is never in memory; ww3 output unchanged apart from `geo` in `dataset.json`.
  - [x] x3d: grid dims and lat/lon names come from `dataset.json` (`dims`, `geo`); tested with a synthetic Y/X store and a real 3-step NorKyst store.
  - [x] `../devops/ppi/update-nk800.sh` and `status-nk800.sh`.
  - [ ] run a full conversion (~169 steps, ~10 GB read from THREDDS, ~2.7 GB store) and publish; check time and memory. Don't stress THREDDS.
  - [ ] lat/lon (float64) are stored in both `map.zarr` and `ts.zarr` (~44 MB each): use float32 or one layout only.
  - [ ] reuse int16 time steps already published in the previous run, to cut the THREDDS load.
  - [ ] benchmark THREDDS vs x3d for NorKyst, add to the README.

- Data-layer split (see `docs/format.md`, `../x3d-publisher`)
  - [x] remove the pipeline from wave-world (`pipeline/{convert,publish,fetch,run.sh}`, `Dockerfile.pipeline`, their tests) once `../devops/ppi/update-ww3.sh` calls `x3d-publish run ww3_4km` instead of `$WAVE_WORLD/pipeline/run.sh`; `serve.py` (dev server) stays with wave-world or moves to x3d.
  - [ ] x3d: check `format_version`. (`dims`/`geo` are read from the metadata; done.)
  - [x] publisher CI: builds a small store and runs x3d's tests against it.
  - [ ] publisher Dockerfile/CronJob.

## usage
- [ ] does local store (x3d+file) work symmetrically to the s3 version?
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

- [ ] eventually we need to decouple the pipeline from wave-world. i think it should live in its own repository. both x3d and wave-world will be users of the produced data (either local or remote/s3).

## norkyst
- [ ] NorKyst-800 (prefix `norkyst800` in the bucket, surface layer only):
  - source: newest `fc` file on THREDDS (`fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.fc.<date>00.nc`, 97 hourly steps, 902x2602, 16 depths; `an` files are 24 steps); the `fc` file is 21 GB, `an` 5.3 GB. Check which date is newest on THREDDS (the catalog listing looked stale).
  - variables at `depth=0`: `temperature`, `salinity`, `u_eastward`, `v_northward`, `zeta`, `Uwind`, `Vwind`; coordinates `lat`/`lon` (2-D), dims `Y`/`X`.
  - estimated store: ~1 GB per layout, ~2 GB with `ts` + `map` per run (about 5x a ww3 run).
  - [ ] wave-world `convert.py`: make it model-driven (dims, variables, scales, overview variables, shards, lat/lon names, depth selection, source file lookup); keep the ww3 output unchanged and its tests passing.
  - [ ] pack to int16 while downloading (float32 for the whole run would be ~6 GB).
  - [ ] x3d: stop assuming `rlat`/`rlon`, `latitude`/`longitude` (`GEO_COORDS`) and the ww3 overview grid; test with a NorKyst-like synthetic store.
  - [ ] `../devops/ppi/update-nk800.sh` (and `status-nk800.sh`), modelled on `update-ww3.sh`: bucket `waves-ww3`, prefix `norkyst800`, same credentials and cron setup.
  - [ ] benchmark THREDDS vs x3d for NorKyst, add to the README.

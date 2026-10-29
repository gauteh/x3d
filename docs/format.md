# x3d store format (draft, version 1)

The contract between producers (`x3d-publisher`) and readers (`x3d`, the web viewer, anything that reads
Zarr v3). Everything is plain files/objects behind HTTP `GET` with `Range`, so a local directory and an S3
bucket are interchangeable. Nothing here needs a server.

## Object layout

```
<store>/latest.json                 {"run": "<run>"}       the only mutable object (short TTL)
<store>/<run>/dataset.json          metadata, see below
<store>/<run>/map.zarr/             field layout  (time=1, whole grid per shard)
<store>/<run>/ts.zarr/              series layout (all time, small spatial tiles)
<store>/<run>/ov.zarr/              overview layout (all time, coarsened grid)
<store>/<run>/grid.bin              decimated lon/lat mesh (viewer geometry; optional for readers)
```

`<run>` is a UTC timestamp `YYYYMMDDTHHMMSSZ`; lexical order is chronological. Everything below `<run>/` is
**immutable** (`Cache-Control: public, max-age=31536000, immutable`); `latest.json` is short-lived
(`max-age=60`). A store prefix is "the dataset": `x3d+https://host/bucket/ww3_4km`.

## Publishing order (atomicity)

1. Data objects of the run, 2. metadata (`zarr.json`, `dataset.json`, `grid.bin`), 3. `latest.json`.
A reader that resolved `latest.json` therefore always sees a complete run. Old runs are pruned
afterwards: the newest N and the one `latest.json` names stay, and runs younger than a grace period are
kept so clients holding a cached `latest.json` do not break. Concurrent publishers are safe (last
`latest.json` wins; runs never overwrite each other).

## Zarr arrays

Zarr v3, consolidated metadata in the group's `zarr.json`. Only standard codecs: `sharding_indexed` with
the **index at the start** (readable with a plain `bytes=0-N` range, no suffix ranges, no object size), inner
codecs `bytes` (little endian) + `zstd`, index codecs `bytes` + `crc32c`.

| layout | dims | shard | inner chunk | use |
|---|---|---|---|---|
| `map.zarr` | (time, y, x) | (1, whole grid) | (1, 256, 256) | one field: one request per time step |
| `ts.zarr`  | (time, y, x) | (all time, 128, 128) | (all time, 16, 16) | a point series: index + one chunk |
| `ov.zarr`  | (time, y, x) | none | (24, whole coarse grid) | quick look at every time step |

`ov.zarr` is at `1/ov_factor` resolution (block means, a cell needs at least half valid samples,
directions averaged as unit vectors) and holds only `ov_vars`. It has no `y`/`x` coordinate variables:
readers derive them from the full-resolution grid.

Data variables are `int16` with attributes `scale_factor` and `_FillValue = -32768` (CF packing); the Zarr
`fill_value` equals `_FillValue`. Chunks that are entirely fill are **absent** (shard index entry
`0xFFFFFFFFFFFFFFFF`), and a missing shard object is all fill. Readers must treat both as fill.
Values are `raw * scale_factor` with a float32 scale.

## dataset.json

| key | meaning |
|---|---|
| `format_version` | integer, 1. Readers should refuse a larger major version they do not know. |
| `name`, `run`, `model_run` | store name, run id, reference time of the newest model run (ISO, may be null) |
| `times` | the time axis, ISO 8601 UTC strings (`...Z`), identical for all layouts |
| `ny`, `nx`, `dims` | grid size and the names of the y and x dimensions: `{"y": "rlat", "x": "rlon"}` |
| `geo` | names of the 2-D latitude/longitude variables, `["lat", "lon"]` (default when absent) |
| `ov_factor`, `ov_tchunk`, `ov_vars` | overview resolution divisor, its time chunk, variables it holds |
| `scales` | `scale_factor` per variable |
| `mesh_stride`, `mesh_ny`, `mesh_nx` | `grid.bin` geometry: lon mesh then lat mesh, float32 LE |
| `crs` | optional: `{grid_mapping, grid_mapping_name, proj4, attrs}`, the source projection (null if the source had none) |
| `attrs`, `variables` | optional: global attributes, and per-variable attributes, as published |
| `bbox` | optional: `[lon_min, lat_min, lon_max, lat_max]` |

### Metadata in the Zarr groups

The projection is relayed as a CF grid-mapping variable: a scalar (0-d, no chunk data, no `dimension_names`)
array in every layout group, named as in the source (`rotated_pole`, `projection_stere`), whose attributes are the
CF projection parameters (plus `proj4` when the source has it). Data variables keep their `grid_mapping`
attribute pointing at it. Readers expose it as a coordinate. Source global attributes (title, institution,
license, ...) and variable attributes (`standard_name`, `long_name`, `units`) are the group/array attributes;
models may override or drop global attributes.

Layout shapes and chunking are also in each array's own `zarr.json`; readers pick a layout from those
(the cost model in `x3d.layout`), not from hard-coded values.

## Not yet specified / open

- Vertical dimensions: the publisher selects one level (NorKyst-800 surface, `select = {depth = 0.0}`); stores
  hold horizontal fields only. `dims` and `geo` name the horizontal dimensions and coordinates.
- A consolidated index sidecar so a warm client needs fewer round trips (see `todo.md`).
- Whether readers or producers own the `ov` y/x coordinates.

## Conformance

The producer's CI builds a small store and runs x3d against it; x3d's own tests use a synthetic store
(`tests/conftest.py`). Both must agree on this document.

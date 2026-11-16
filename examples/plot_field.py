"""Open the wave store over plain HTTP with xarray and plot a field and a point time series.

    python examples/plot_field.py [store_url] [output.png]

Needs: xarray, zarr>=3, fsspec, aiohttp, matplotlib. Only the chunks that are needed are downloaded.
"""
import json
import sys
import urllib.request

import matplotlib.pyplot as plt
import xarray as xr

root = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/ww3_4km"
out = sys.argv[2] if len(sys.argv) > 2 else None

# latest.json names the current immutable run
run = json.load(urllib.request.urlopen(f"{root}/latest.json"))["run"]
print("run", run)

# full-resolution fields (one shard per time step) and the time-series layout (one shard per spatial block)
field = xr.open_zarr(f"{root}/{run}/map.zarr")
series = xr.open_zarr(f"{root}/{run}/ts.zarr")
print(field)

# combined sea (hs), wind sea (phs0) and swell (phs1) at the first time step; decoded to metres, land is NaN
t = field.time.values[0]
sel = field[["hs", "phs0", "phs1"]].sel(time=t).load()

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, (name, title) in zip(axes, [("hs", "Combined sea"), ("phs0", "Wind sea"), ("phs1", "Swell")]):
    # rotated-pole grid: plotted against the grid axes (longitude/latitude are 2-D coordinates)
    sel[name].plot(ax=ax, vmin=0, vmax=8, cmap="viridis", cbar_kwargs={"label": "Hs (m)"})
    ax.set_title(f"{title}\n{str(t)[:16]}Z")
fig.tight_layout()

if out is not None:
    fig.savefig(out, dpi=110)
    print("wrote", out)
else:
    plt.show()

# time series at the grid point nearest to 65N, 5E (Norwegian Sea)
lon, lat = 5.0, 65.0
d2 = (series.longitude - lon) ** 2 + (series.latitude - lat) ** 2
iy, ix = (int(i) for i in d2.argmin(dim=("rlat", "rlon")).values())
pt = series[["hs", "phs0", "phs1"]].isel(rlat=iy, rlon=ix).load()
fig, ax = plt.subplots(figsize=(9, 3.5))
for name, label in [("hs", "Combined"), ("phs0", "Wind sea"), ("phs1", "Swell")]:
    pt[name].plot(ax=ax, label=label)
ax.set_ylabel("Hs (m)")
ax.set_title(f"{float(series.latitude[iy, ix]):.2f}N {float(series.longitude[iy, ix]):.2f}E")
ax.legend()
fig.tight_layout()

if out is not None:
    series_out = out.replace(".png", "_series.png")
    fig.savefig(series_out, dpi=110)
    print("wrote", series_out)
else:
    plt.show()

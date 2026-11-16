"""Sentinel-3A altimeter along-track series: open the published store, print statistics and plot.

    python examples/altimeter_s3a.py [--start 2026-08-01] [--end 2026-09-01] [--plot out.png]

The store is one 1-D series on `time` (with `latitude`/`longitude` as coordinates along the track); the layout is
detected by x3d. Only the chunks covering the selected period are read.
"""
import argparse

import numpy as np
import xarray as xr

URL = "x3d+https://rgw.met.no/079d3efebae2494db2a13c49766989ab:altimeter/altimeter_s3a"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=URL)
    p.add_argument("--start", default="2026-08-01")
    p.add_argument("--end", default="2026-09-01")
    p.add_argument("--plot", metavar="FILE", help="write a figure (needs matplotlib)")
    a = p.parse_args()

    ds = xr.open_dataset(a.url)
    print(f"{ds.attrs.get('title', '')}: {ds.sizes['time']:,} samples, {ds.time.values[0]} .. {ds.time.values[-1]}")

    sub = ds[["VAVH", "WIND_SPEED"]].sel(time=slice(a.start, a.end)).load()
    hs = sub.VAVH
    print(f"\n{a.start} .. {a.end}: {sub.sizes['time']:,} samples, {int(hs.notnull().sum()):,} with Hs")
    print(f"Hs [m]  mean {float(hs.mean()):.2f}  median {float(hs.median()):.2f}  p95 {float(hs.quantile(0.95)):.2f}  max {float(hs.max()):.2f}")

    daily = hs.resample(time="1D").mean()
    print("\ndaily mean Hs (m):")
    print(daily.to_series().round(2).to_string(max_rows=10))

    box = {"North Sea": (-4, 9, 51, 62), "Nordic seas": (-20, 20, 62, 76)}
    for name, (w, e, s, n) in box.items():
        sel = (sub.longitude >= w) & (sub.longitude <= e) & (sub.latitude >= s) & (sub.latitude <= n)
        x = hs.where(sel)
        print(f"{name:12s} n={int(x.notnull().sum()):>7,}  mean Hs {float(x.mean()):.2f} m  p95 {float(x.quantile(0.95)):.2f} m")

    if a.plot:
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1.4]})
        sc = ax0.scatter(sub.longitude, sub.latitude, c=hs, s=1, vmin=0, vmax=np.nanpercentile(hs, 99), cmap="viridis")
        ax0.set(xlabel="longitude", ylabel="latitude", title=f"VAVH along the tracks, {a.start} .. {a.end}")
        fig.colorbar(sc, ax=ax0, label="significant wave height (m)")
        ax1.plot(daily.time, daily, lw=1)
        ax1.set(ylabel="daily mean Hs (m)", title="all tracks")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(a.plot, dpi=130)
        print(f"\nwrote {a.plot}")


if __name__ == "__main__":
    main()

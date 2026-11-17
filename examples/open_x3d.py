"""Minimal x3d example: open the public wave forecast and read a point series and a field.

Needs xarray and x3d (https://github.com/gauteh/x3d).
"""
import xarray as xr

# the "x3d+" prefix selects the engine; data is lazy, only what is read is downloaded
ds = xr.open_dataset("x3d+https://rgw.met.no/079d3efebae2494db2a13c49766989ab:waves-ww3/ww3_4km")
print(ds)

# significant wave height [m] at one grid point, all forecast hours
print(ds.hs.isel(rlat=500, rlon=300).to_series())

# the whole field at one time step
field = ds.hs.isel(time=10).load()
print(float(field.max()), "m max")

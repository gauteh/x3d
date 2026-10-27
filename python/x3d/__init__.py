"""xarray backend for the waves.met.no Zarr data store (``x3d+https://.../ww3_4km``).

The store keeps, per model run, the same data in several layouts (see ``wave-world``):

``ts``   shards ``(all time, 128, 128)``: cheap point time series
``map``  one shard per time step: cheap fields
``ov``   block-mean 1/4 resolution: cheap overview, lossy so only used on request

A :class:`~x3d.backend.X3DArray` picks between the exact layouts per access.
"""

from x3d.backend import X3DBackendEntrypoint, open_x3d

__all__ = ["X3DBackendEntrypoint", "open_x3d"]

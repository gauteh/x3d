"""The usage examples of the README, kept in sync by hand, run against the public RGW store."""
import pytest
import xarray as xr

pytestmark = pytest.mark.rgw

WW3 = "https://rgw.met.no/079d3efebae2494db2a13c49766989ab:waves-ww3/ww3_4km"
NORKYST = "https://rgw.met.no/079d3efebae2494db2a13c49766989ab:norkyst800m-v3/norkyst800"


def test_readme_usage(rgw_url):
    ds = xr.open_dataset("x3d+" + WW3)
    assert ds.hs.isel(rlat=500, rlon=300).values.size > 0
    assert ds.hs.isel(time=10).values.size > 0
    ov = xr.open_dataset("x3d+" + WW3, layout="ov")
    assert ov.hs.isel(time=0).size > 0


def test_norkyst():
    ds = xr.open_dataset("x3d+" + NORKYST)
    assert ds.u_eastward.isel(time=0).size > 0

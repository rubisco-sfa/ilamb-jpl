"""
- Especially since your resolution is quite coarse, do you have a variable which
  represents the underlying land fractions? 
- Are the output quantities averaged over land or over the cell area?
- We will need the same title, version, etc. for the attributes.
"""

import cf_xarray  # noqa
import numpy as np
import xarray as xr
import cftime as cf


def convert_cardamom(ds: xr.Dataset, vname: str) -> None:

    # We will build up the converted dataset in here.
    out = {}

    # Is it ok to rename 'time_fluxes' to 'time'?
    ds = ds.rename(dict(time_fluxes="time"))

    # A convenience assignment
    var = ds[vname]

    # In ILAMB, we can't make direct use of quantiles. I propose that we use the 50%
    # quantile as the primary result.
    out[vname] = var.sel(quantile=0.5).drop_vars("quantile")

    # I understand why carbon units are important, but they aren't CF-compliant :/
    out[vname].attrs["units"] = out[vname].attrs["units"].replace("gC", "g")

    # We do have some methodology to handle reference data uncertainty as a scalar field
    # in time and space. I propose that we use a geometric mean of the 75% - 50% and the
    # 25% - 50% quantiles as a measure of this uncertainty. Maybe this is not an
    # appropriate interpretation?
    out[f"{vname}_uncert"] = np.sqrt(
        (var.sel(quantile=0.75) - var.sel(quantile=0.5)) ** 2
        + (var.sel(quantile=0.25) - var.sel(quantile=0.5)) ** 2
    )

    # Add some metadata here to reflect the units and relationship to the main variable
    out[vname].attrs["ancillary_variables"] = f"{vname}_uncert"
    out[f"{vname}_uncert"].attrs = {
        "long_name": f"{vname} standard_error",
        "units": out[vname].attrs["units"],
    }

    # Your times are fine, but ILAMB was having trouble with the datetime64 resolution.
    # I am rewriting them to 'noleap' and adding bounds which helps ILAMB in comparing
    # to other sources that use different calendars.
    out = xr.Dataset(out)
    out["time"] = [cf.DatetimeNoLeap(t.dt.year, t.dt.month, 15) for t in out["time"]]
    out["time_bounds"] = xr.DataArray(
        np.asarray(
            [
                [cf.DatetimeNoLeap(t.dt.year, t.dt.month, 1) for t in out["time"]],
                [
                    cf.DatetimeNoLeap(
                        (t.dt.year + 1) if t.dt.month == 12 else t.dt.year,
                        1 if t.dt.month == 12 else (t.dt.month + 1),
                        1,
                    )
                    for t in out["time"]
                ],
            ],
        ).T,
        dims=("time", "nb"),
    )

    # As a reference product, in ILAMB we provide the following information in order for
    # users to understand what the data represent.
    out.attrs = {
        "title": "",
        "version": "",
        "institution": "",
        "source": "",
        "references": "",
    }
    out.to_netcdf(
        f"CARDAMOM_{vname}.nc",
        encoding={
            "time": {"units": "days since 1850-01-01", "bounds": "time_bounds"},
            "time_bounds": {"units": "days since 1850-01-01"},
            vname: {"zlib": True},
        },
    )


if __name__ == "__main__":

    convert_cardamom(xr.open_dataset("_raw/GPP4ILAMBv1.nc"), "gpp")

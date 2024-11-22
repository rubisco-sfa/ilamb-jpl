import cftime as cf
import ilamb3.dataset as dset
import intake
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from intake_esgf import ESGFCatalog


def cf_compliant(ds: xr.Dataset) -> xr.Dataset:
    """Convert the dataset to CF-compliant file that ILAMB can use."""
    # Time has not been encoded correctly, but we can use the values to encode it
    # properly.
    t = np.array(
        [cf.DatetimeNoLeap(y, m, 15) for (y, m, _) in ds["start_date"]],
    )
    tb = np.array(
        [
            [cf.DatetimeNoLeap(y, m, d) for (y, m, d) in ds["start_date"]],
            [
                cf.DatetimeNoLeap(
                    (y + 1) if m == 12 else y, 1 if m == 12 else (m + 1), d
                )
                for (y, m, d) in ds["start_date"]
            ],
        ]
    ).T
    ds = (
        ds.assign_coords({"time": t})
        .rename({"n_months": "time"})
        .drop_vars("start_date")
    )

    # We need to add an `units` attribute in each variable. Based on our email exchange,
    # the units are grams per square meter per year.
    for var in ds:
        ds[var].attrs["units"] = "g m-2 year-1"
    ds["time_bounds"] = (("time", "nb"), tb)

    # Let's rename to CMOR variable names (compatible with CMIP experiments).
    ds = ds.rename(
        {key: key.replace("land", "nbp").replace("ocean", "fgco2") for key in ds}
    )

    # Since your data comes with std, let's encode that as well. ILAMB can use this as a
    # measure of uncertainty.
    for var in ds:
        std = f"{var}_std"
        if std in ds:
            ds[var].attrs["ancillary_variables"] = f"{var}_std"
            ds[std].attrs["standard_name"] = f"{var} standard_error"
    return ds.pint.quantify()


def plot_global_totals():
    """The following produces `diag2.png` which I used to make sure I understood the
    sign conventions of the JPL dataset.
    """
    ds = xr.open_dataset("EnsMean_gridded_fluxes_2015-2020.nc4")
    fig, ax = plt.subplots(tight_layout=True)
    ds["residual"] = ds["land"] + ds["ocean"] - ds["net"]
    for var in ["land", "ocean", "net", "residual"]:
        ds[var].attrs["units"] = "g m-2 year-1"
        v = dset.convert(dset.integrate_space(ds, var), "Pg yr-1")
        v.plot(ax=ax, label=var)

    ax.set_title("global sum")
    fig.legend()
    fig.savefig("diag2.png")
    plt.close()


def plot_histogram():
    """The following produces `diag1.png` which is meant to plot a histogram of values
    of the JPL carbon product and compare to CESM2. We used this to understand units
    differences.
    """

    def mask_outliers(
        ds: xr.Dataset, low: float = 0.02, high: float = 0.98
    ) -> xr.Dataset:
        return xr.where((ds > ds.quantile(low)) & (ds < ds.quantile(high)), ds, np.nan)

    ref = xr.open_dataset("EnsMean_gridded_fluxes_2015-2020.nc4")
    cat = ESGFCatalog().search(
        experiment_id="historical",
        source_id="CESM2",
        variable_id="nbp",
        variant_label="r1i1p1f1",
    )
    mod = cat.to_dataset_dict()["nbp"]
    mod = mod.isel({"time": slice(-73, -1)})
    mod = -mod["nbp"].pint.quantify().pint.to("g m-2 year-1")

    # mask outliers
    ref = mask_outliers(ref)
    mod = mask_outliers(mod)

    fig, axs = plt.subplots(figsize=(10, 5), ncols=2, tight_layout=True)
    ref["land"].plot(ax=axs[0])
    mod.plot(ax=axs[1])

    axs[0].set_title("EnsMean['land'] histogram (outliers removed)")
    axs[1].set_title("-CESM2 nbp histogram (outliers removed)")
    fig.savefig("diag1.png")
    plt.close()


def plot_vs_Hoffman(ds: xr.Dataset):
    hof = (
        intake.open_catalog(
            "https://raw.githubusercontent.com/nocollier/intake-ilamb/main/ilamb.yaml"
        )["nbp | Hoffman"]
        .read()
        .pint.quantify()
    )
    fig, ax = plt.subplots(tight_layout=True)
    for var in ["nbp", "fgco2"]:
        v = dset.convert(dset.integrate_space(ds, var), "Pg yr-1")
        v = v.pint.dequantify()
        v = -v.groupby("time.year").mean()
        v.plot(ax=ax, label=f"JPL {var}")
    for var in ["nbp", "fgco2"]:
        v = hof[var].groupby("time.year").mean()
        v.plot(ax=ax, label=f"Hoffman {var}")
    ax.set_title("Global Totals")
    fig.legend(loc=2)
    fig.savefig("global_totals.png")


if __name__ == "__main__":

    # Read in the JPL outputs and make CF-compliant
    jpl = cf_compliant(
        xr.merge(
            [
                xr.open_dataset("EnsMean_gridded_fluxes_2015-2020.nc4"),
                xr.open_dataset("EnsStd_gridded_fluxes_2015-2020.nc4")
                .rename({key: f"{key}_std" for key in ["ocean", "land", "net"]})
                .drop_vars("start_date"),
            ]
        )
    )
    jpl.pint.dequantify().to_netcdf(
        "OCO_carbon_fluxes.nc", encoding={key: {"zlib": True} for key in jpl}
    )

    plot_vs_Hoffman(jpl)

import os
import time

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

    download_stamp = time.strftime(
        "%Y-%m-%d",
        time.localtime(os.path.getmtime("EnsMean_gridded_fluxes_2015-2020.nc4")),
    )
    generate_stamp = time.strftime("%Y-%m-%d")

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
    jpl.attrs = {
        "title": "OCO-2-V10-MIP",
        "version": "1",
        "institutions": "Jet Propulsion Laboratory, California Institute of Technology, Colorado State University, NASA Goddard Space Flight Center, Global Modeling and Assimilation Office, Greenbelt, MD, USA, School of Mathematics and Applied Statistics, University of Wollongong, Wollongong, NSW, Australia, Laboratoire des Sciences du Climat et de L'Environnement, LSCE/IPSL, CEA-CNRS-UVSQ, Université Paris-Saclay, 91191 Gif-sur-Yvette, France, Department of Physics, University of Toronto, Toronto, Ontario, Canada, Satellite Observation Center, Earth System Division, National Institute for Environmental Studies, Tsukuba, Japan, Department of Environmental Health and Engineering, Johns Hopkins University, Baltimore, MD, USA, Centre for Atmospheric Sciences, Indian Institute of Technology Delhi, New Delhi, India, Department of Atmospheric and Oceanic Science, University of Maryland, College Park, MD, USA, Laboratory of Numerical Modeling for Atmospheric Sciences & Geophysical Fluid Dynamics, Institute of Atmospheric Physics, Chinese Academy of Sciences, Beijing, China, NOAA Global Monitoring Laboratory, Boulder, CO, USA",
        "sources": "Ensemble mean of 13 top-down inversion models assimilating both in situ and land nadir and land glint satellite CO2 observations from OCO-2.",
        "history": f"""
{download_stamp}: downloaded files from https://gml.noaa.gov/ccgg/arc/?id=150;
{generate_stamp}: converted to ILAMB-ready netCDF""",
        "references": """
@ARTICLE{OCO2,
  author = {Byrne, B., Baker, D. F., Basu, S., Bertolacci, M., Bowman, K. W., Carroll, D., Chatterjee, A., Chevallier, F., Ciais, P., Cressie, N., Crisp, D., Crowell, S., Deng, F., Deng, Z., Deutscher, N. M., Dubey, M. K., Feng, S., García, O. E., Griffith, D. W. T., Herkommer, B., Hu, L., Jacobson, A. R., Janardanan, R., Jeong, S., Johnson, M. S., Jones, D. B. A., Kivi, R., Liu, J., Liu, Z., Maksyutov, S., Miller, J. B., Miller, S. M., Morino, I., Notholt, J., Oda, T., O'Dell, C. W., Oh, Y.-S., Ohyama, H., Patra, P. K., Peiro, H., Petri, C., Philip, S., Pollard, D. F., Poulter, B., Remaud, M., Schuh, A., Sha, M. K., Shiomi, K., Strong, K., Sweeney, C., Té, Y., Tian, H., Velazco, V. A., Vrekoussis, M., Warneke, T., Worden, J. R., Wunch, D., Yao, Y., Yun, J., Zammit-Mangion, A., and Zeng, N.},
  title= {National CO2 budgets (2015-2020) inferred from atmospheric CO2 observations in support of the global stocktake},
  journal = {Earth Syst. Sci. Data},
  volume = {15},
  year = {2023},
  page = {963-1004},
  doi = {https://doi.org/10.5194/essd-15-963-2023}
}
""",
    }
    jpl.pint.dequantify().to_netcdf(
        "OCO_carbon_fluxes.nc", encoding={key: {"zlib": True} for key in jpl}
    )

    plot_vs_Hoffman(jpl)

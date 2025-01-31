import matplotlib.pyplot as plt
import xarray as xr
from ilamb3.analysis import bias_analysis
from ilamb3.dataset import convert
from intake_esgf import ESGFCatalog

ref = xr.open_dataset("CARDAMOM_carbon_fluxes.nc")
ref = convert(ref, "g m-2 d-1", "nbp")
ref["nbp"] = -ref["nbp"]
ref["nbp_std"] = -ref["nbp_std"]

cat = (
    ESGFCatalog()
    .search(
        experiment_id=["ssp585"],
        variable_id=["nbp"],  # , "fgco2"],
        source_id="CanESM5",
        frequency="mon",
    )
    .remove_ensembles()
)
mod = cat.to_dataset_dict(ignore_facets=["table_id"])
com = mod[list(mod.keys())[0]]

analysis = bias_analysis("nbp")
df, ds_ref, ds_com = analysis(ref, com)
ds_com = ds_com.compute()

fig, axs = plt.subplots(nrows=2, ncols=2, tight_layout=True)
vmin = min(ds_ref["mean"].quantile(0.02), ds_com["mean"].quantile(0.02)).values
vmax = max(ds_ref["mean"].quantile(0.98), ds_com["mean"].quantile(0.98)).values
vmax = max(-vmin, vmax)
vmin = -vmax
ds_ref["mean"].plot(ax=axs[0, 0], cmap="PiYG", vmin=vmin, vmax=vmax)
ds_com["mean"].plot(ax=axs[0, 1], cmap="PiYG", vmin=vmin, vmax=vmax)
ds_com["bias"].plot(ax=axs[1, 0])
ds_com["bias_score"].plot(ax=axs[1, 1], cmap="plasma", vmin=0, vmax=1)
axs[0, 0].set_title("CARDAMOM land")
axs[0, 1].set_title("CanESM5 ssp585 nbp")
axs[1, 0].set_title("nbp bias")
axs[1, 1].set_title("ILAMB bias score")
plt.show()

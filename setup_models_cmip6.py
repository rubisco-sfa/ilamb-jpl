import matplotlib as mpl
import yaml
from intake_esgf import ESGFCatalog

cat = ESGFCatalog().search(
    experiment_id="historical",
    source_id=[
        "BCC-CSM2-MR",
        "CanESM5",
        "CESM2",
        "GFDL-ESM4",
        "IPSL-CM6A-LR",
        "MIROC-ESM2L",
        "MPI-ESM1.2-HR",
        "NorESM2-LM",
        "UKESM1-0-LL",
    ],
    variable_id=["areacella", "sftlf"],
    frequency=["fx"],
)

# How many of our variables does each model group (unique combination of
# ('source_id','member_id','grid_label')) have?
counts = cat.model_groups()


# We use this function to remove model groups that have less than the max
def has_max_counts(df) -> bool:
    model = df.iloc[0]["source_id"]
    if len(df) == counts[counts.index.get_level_values(0) == model].max():
        return True
    return False


# Then we also select just the 'smallest' member_id that has all of our variables.
cat.remove_incomplete(has_max_counts).remove_ensembles()


paths = cat.to_path_dict(ignore_facets=["institution_id"])
paths = {
    model: {
        key: list(set([str(p.parent) for p in data]))
        for key, data in paths.items()
        if key.startswith(model)
    }
    for model in cat.df.source_id.unique()
}

# Define some model colors using a matplotlib color sequence
clrs = mpl.color_sequences.get("tab10")
colormap = {
    model: "#%02x%02x%02x"
    % (int(255 * clrs[i][0]), int(255 * clrs[i][1]), int(255 * clrs[i][2]))
    for i, model in enumerate(cat.df.source_id.unique())
}

# Output the ILAMB model setup
paths = {
    model: {
        "modelname": model,
        "color": colormap[model],
        "path": None,
        "paths": sorted([k for _, key in keys.items() for k in key]),
    }
    for model, keys in paths.items()
}
with open("models_cmip6.yaml", mode="w") as out:
    out.write(yaml.dump(paths))

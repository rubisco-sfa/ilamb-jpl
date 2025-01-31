import glob
from pathlib import Path

import matplotlib as mpl
import yaml

ROOT = "/home/nate/work/ilamb-jpl"

models = sorted(
    list(
        set(
            [
                Path(f).stem.split("_")[0]
                for f in glob.glob(str(Path(ROOT) / "models/TRENDY/*.nc"))
            ]
        )
    )
)

# Define some model colors using a matplotlib color sequence
clrs = mpl.color_sequences.get("tab20")
colormap = {
    model: "#%02x%02x%02x"
    % (int(255 * clrs[i][0]), int(255 * clrs[i][1]), int(255 * clrs[i][2]))
    for i, model in enumerate(models)
}

paths = {
    model: {
        "modelname": model,
        "filter": model,
        "color": colormap[model],
        "path": f"{ROOT}/models/TRENDY/",
    }
    for model in models
}
with open("models_trendy.yaml", mode="w") as out:
    out.write(yaml.dump(paths))


mpirun -n 2 ilamb-run \
--config jpl.cfg \
--model_setup models_trendy.yaml \
--define_regions GlobalLand.nc Koppen.nc \
--regions global tropical arid temperate cold \
--title ILAMB-JPL

# rnet

Code-only distribution of the NN4R map-simulation and inference package.
The original repository and saved scientific products are unchanged.

Install with `pip install -e .`; choose optional dependencies as needed,
for example `pip install -e '.[simulation,neural]'`.

Run `notebooks/quickstart.ipynb` for basic API usage. Simulation requires
user-supplied `base/data/` inputs (`camb.ini`, hits map, binary and apodized
masks, padded mask). No simulation input files, maps, trained weights,
configuration files, results, or cluster-specific job scripts are included.

The `pyproject.toml` file is packaging metadata needed for installation;
it is not a simulation or training configuration.

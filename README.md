# Infer periodic timeseries parameters using Chainsail

This repository contains scripts, experiments and other efforts to infer parameters of the periodic timeseries model discussed in [this Stan Discourse thread](https://discourse.mc-stan.org/t/ideas-for-modelling-a-periodic-timeseries/22038).
The model is highly multimodal and thus is a potential good usecase for Chainsail's Replica Exchange algorithm.
For now, this repository contains
- `probability.py`: a Python module casting the above model into forms consumable by Chainsail, namely:
  - a wrapper around the Stan-defined model using the [Stan wrapper](https://github.com/tweag/chainsail-resources/blob/main/chainsail_helpers/chainsail_helpers/pdf/stan/__init__.py) from the `chainsail-helpers` package
  - a from-scratch implementation of the posterior using pure NumPy / Python, which has the advantage that it provides separate `log_likelihood` and `log_prior` methods that allow for the use of the [likelihood tempering scheme](https://github.com/tweag/chainsail-resources/blob/main/documentation/algorithms/replica_exchange.md#likelihood-tempering). Unfortunately, I'm not sure it is correct, as I got confused with the bounded / unbounded variable and density transforms.
- `chainsail_compatible_model.stan`: a slightly modified version of the original Stan code in the above Discourse post that is compatible with Chainsail (Chainsail requires a flat parameter array, while the original Stan model takes jagged parameters)
- `data.csv`: test data generated with the scripts provided in the [original thread author's gist ](https://gist.github.com/mike-lawrence/716973647a9656133c49e012f4547103)
- `analyze_stan_model_results.py`: a Python script that plots histograms and traces for all (transformed) parameters as well as a 2D histogram of phase vs. frequency. It expects a single `.npy` file with all samples as its only command line argument; after a Chainsail run, this file can be obtained using the `concatenate-samples` script from the `chainsail-helpers` package.
- `sampling_results.png`: an example figure of results of a Chainsail run that uses the `httpstan`-backed model definition

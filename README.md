# Efficient Quadratic Corrections for Frank-Wolfe Algorithms

In this project my personal contribution was to implement the SOCGS algorithm and to conduct the regression experiments on the `gisette` data set.

## Original README from the Supplementary Material from [here](https://openreview.net/forum?id=Sz2i4dwDem)
Source code for the submission.

The library code is in `src/`, the individual experiment scripts are in `experiments/`.

- `tensor_completion.jl` contains the LMO, function and gradient implementations for the tensor completion experiment. The experiment script is `tensor_experiment.jl`.
- `src/socgs_algorithm` contains the implementation of the SOCGS algorithm and `src/load_gisette.jl` is the script used in `experiments/socgs_logistic_regression_experiment.jl` to load the `gisette` data set. To reproduce the SOCGS experiments found in the paper:
  * download the `gisette dataset` [here](https://archive.ics.uci.edu/static/public/170/gisette.zip);
  * extract `gisette_train.data` and `gisette_train.label` and put them in a directory `experiments/socgs_data`;
  * run `experiments/socgs_logistic_regression_experiment.jl` with different boolean values for `do_pvm_with_quadratic_active_set` and `do_wolfe`;
  * use `experiments/socgs_logistic_regression_plots.jl` to plot the trajectories.
- `src/entanglement_detection.jl` contains the LMO and a wrapper for solving entanglement detection problem with FW methods. The experiment for the Horodecki state is given in `entanglement_detection_experiment.jl`.
- `utils.jl`contains function for comparing the quadratic correction methods with BPCG on a given problem. The regression problem over the K-Sparse polytope and the Birkhoff projection problem are given in `ķsparse_regression_experiment.jl`and `birkhoff_projection_experiment.jl`, respectively.
- `alm_utils.jl`contains functions for comparing running the ALM and SCG algorithms with different corrective steps in a block-cyclic manner. The experiments for the intersection problem using ALM are contained in `alm_experiment.jl`and the experiments for the projection problem over the intersection using SCG are in `splitting_experiment.jl`.
  

using FrankWolfe
using LinearAlgebra
using Random

import HiGHS
import MathOptInterface as MOI

include("../src/utils.jl")
include(joinpath(dirname(pathof(FrankWolfe)), "../examples/plot_utils.jl"))


function build_problem(n, seed)

    Random.seed!(seed)

    xp = rand(n, n)
    normxp2 = dot(xp, xp)

    a = 1 / n^2

    f(x) = (normxp2 - 2dot(x, xp) + dot(x, x)) * a

    function grad!(storage, x)
        storage .= 2 * (x - xp) * a
    end

    # BirkhoffPolytopeLMO via Hungarian Method
    lmo = FrankWolfe.BirkhoffPolytopeLMO()

    # initial direction for first vertex
    direction_mat = ones(n, n)
    x0 = FrankWolfe.compute_extreme_point(lmo, direction_mat)

    A = 2*I * a
    b = -2*xp * a

    return f, grad!, lmo, x0, A, b
end

# Experiment settings
n = 500
seed = 1

start_time = 20 # Interval length N in the paper
scaling_factor = 1.0

trajectories = run_qc_comparison(build_problem, (n, seed), start_time, scaling_factor; max_iteration = 1e5, verbose = true, timeout = 3600)

plot_trajectories(trajectories, ["BPCG" "QC-MNP" "QC-LP"], marker_shapes = [:circle, :star5, :diamond], reduce_size= true, filename = "birkhoff_projection_experiment_$(n)_$(seed).pdf")
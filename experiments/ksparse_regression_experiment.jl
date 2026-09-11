using FrankWolfe
using LinearAlgebra
using Random

import HiGHS
import MathOptInterface as MOI

include("../src/utils.jl")
include(joinpath(dirname(pathof(FrankWolfe)), "../examples/plot_utils.jl"))

function build_problem(n, m, k, τ, seed)
    Random.seed!(seed)
    A = randn(m, n)
    AA = A' * A
    b = randn(m)
    f(x) = 1/m * norm(A*x - b)^2
    function grad!(storage, x)
        storage .= 2 * (AA*x - A'*b) / m
    end
    lmo = FrankWolfe.KSparseLMO(k, τ)
    x0 = FrankWolfe.compute_extreme_point(lmo, ones(n))
    quad_term = 2*AA
    linear_term = -2*A'*b
    return f, grad!, lmo, x0, quad_term, linear_term
end

kwargs = Dict{Symbol, Any}()
kwargs[:verbose] = false
kwargs[:epsilon] = 1e-7
kwargs[:max_iteration] = 1e5
kwargs[:timeout] = 900

# Experiment settings
start_time = 10
scaling_factor = 1.0

n = 500
m = 10000
k = 5
τ = 1.0
seeds = 1
trajectories = run_qc_comparison(build_problem, (n, m, k, τ, seed), start_time, scaling_factor; max_iteration = 1e2, verbose = true, timeout = 3600)

plot_trajectories(trajectories, ["BPCG" "QC-MNP" "QC-LP"], marker_shapes = [:circle, :star5, :diamond], reduce_size= true, filename = "ksparse_regression_experiment_$(n)_$(m)_$(k)_$(τ)_$(seed).pdf")
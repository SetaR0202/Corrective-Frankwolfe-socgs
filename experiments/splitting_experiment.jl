using FrankWolfe
using LinearAlgebra
using Random

import HiGHS
import MathOptInterface as MOI

include("../src/alm_utils.jl")
include(joinpath(dirname(pathof(FrankWolfe)), "../examples/plot_utils.jl"))


# Build the problem for given parameters
function build_problem(n, q, r, c, _)
    lmo1 = FrankWolfe.BirkhoffPolytopeLMO()
    lmo1 = FrankWolfe.TrackingLMO(lmo1)

    m = -n*log(1-q)
    directions = [rand(n, n) for _ in 1:m]
    vs = [FrankWolfe.compute_extreme_point(lmo1, direction) for direction in directions]
    direction = sum(directions)
    direction *= r / sqrt(sum(abs2, direction))
    shift = Matrix(sum(vs) / m) - c*direction

    lmo2 = ShiftedLMO(FrankWolfe.LpNormLMO{Float64,2}(r), shift)

    x01 = FrankWolfe.compute_extreme_point(lmo1, randn(n, n))
    x02 = FrankWolfe.compute_extreme_point(lmo2, randn(n, n))

    Y = rand(n, n)
    a = 1 / n^2
    
    f(x) = a * sum(((x - Y)) .^ 2)
    
    function grad!(storage, x)
        storage .= a * 2 * (x - Y)
    end

    quad_matrix = I
    quad_factor(l) = 2*l*a + 1
    linear_term(l, z) = - 2*a * l * Y - z

    return f, grad!, lmo1, lmo2, x01, x02, linear_term, quad_matrix, quad_factor
end

# Problem settings
n = 500
q = 0.1
r = 1
c = 0.9
seed = 1
params = (n, q, r, c, seed)


λ0 = 1.0
lambda_func = lambda_func_new(λ0)

N = 1

trajectories = run_qc_comparison_alm(build_problem, params, lambda_func; start_time=N, line_search=FrankWolfe.Secant(), verbose=true, max_iteration=1e2, print_iter=1e2, timeout=2000)

plot_trajectories(trajectories, ["BPCG" "QC-MNP" "QC-LP"], marker_shapes = [:circle, :star5, :diamond], reduce_size=false, filename = "splitting_experiment_$(n)_$(seed).pdf")
